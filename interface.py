import customtkinter as ctk
import tkinter.filedialog as fd
import threading
import tempfile
import time
import shutil
import torch
import os
import json

from tkVideoPlayer import TkinterVideo

from pose_analyzer import PoseAnalyzer
from temporal_gcn import TemporalGCNClassifier
from exercise_dataset import LABEL_MAP

from PIL import Image

# Configuração inicial
ctk.set_appearance_mode    ("dark")
ctk.set_default_color_theme("blue")

def get_exercise_data(label):
    
    data = None
    
    if label == "Squat" or label == "Squat_wrong":
        data = [ "Agachamento com Barra", 
"""✅ Execução Correta:\n
- Mantenha os pés paralelos e alinhados à largura dos ombros.
- Realize o movimento com cadência adequada, sem perder o equilíbrio do quadril ou as curvaturas da coluna.
- Certifique-se de que os joelhos sigam a direção dos pés.
- Realize máxima amplitude sem perder as curvaturas fisiológicas da coluna.
- Mantenha a coluna em posição neutra, evitando curvaturas excessivas.
- Distribua o peso entre calcanhares e meio dos pés, com os calcanhares sempre apoiados no chão.""",
"""❌ Erros Comuns:\n
- Valgismo: joelhos colapsando para dentro.
- Calcanhares se levantando: sinal de peso mal distribuído ou mobilidade limitada.
- Hiperextensão ou flexão lombar: curvatura inadequada da coluna durante o movimento.
- Amplitude insuficiente: agachamento muito alto, sem atingir a profundidade necessária.
- Inclinação excessiva do tronco: jogar o peso para frente, sobrecarregando a lombar.""",
            "front\\examples\\squat.mp4", "front\\examples\\squat_output.mp4" ]
        
    elif label == "Deadlift" or label == "Deadlift_wrong":
        data = [ "Levantamento Terra com Barra",
"""✅ Execução Correta:\n
- Faça ativação escapular e mantenha a coluna em posição neutra durante todo o movimento.
- Realize o movimento com cadência adequada e controlada. 
- Os quadris e ombros devem subir simultaneamente, garantindo uma tração eficiente e segura.
- Realize uma boa ativação muscular e simétrica na barra.
- A barra deve permanecer próxima ao corpo.
- Finalize o movimento com a tripla extensão (tornozelo, joelhos e quadril). Sem extensões excessivas.""",
"""❌ Erros Comuns:\n
- Curvar a coluna lombar durante a subida.
- Puxar o peso predominantemente com a lombar, em vez de envolver as pernas e glúteos.
- Deixar a barra distante do corpo, criando uma alavanca desnecessária que sobrecarrega a lombar.
- Não finalizar o movimento com máxima extensão de quadril.""",
            
            "front\\examples\\deadlift.mp4", "front\\examples\\deadlift_output.mp4" ]
        
    elif label == "Benchpress" or label == "Benchpress_wrong":
        data = [ "Supino Reto com Barra",
"""✅ Execução Correta:\n
- Realizar ativação escapular e muscular para gerar boa simetria no movimento. 
- Os cotovelos devem estar em um ângulo confortável, geralmente entre 45º e 75º em relação ao tronco.
- Realize o movimento com cadência adequada e controlada.
- Mantenha os pés firmemente apoiados no chão, contribuindo para a estabilidade corporal e transferindo força para a barra.
- Respeitar as curvaturas fisiológicas da coluna para proporcionar maior segurança.""",
"""❌ Erros Comuns:\n
- Cotovelos excessivamente abertos ou fechados, comprometendo a ativação muscular e aumentando o risco de lesões articulares.
- Hiperlordose lombar, criando uma ponte que sobrecarregue a região.
- Descer a barra rapidamente ou de forma descontrolada, gerando instabilidade e risco.
- Executar o movimento com pouca amplitude.
- Movimento assimétrico, com um lado da barra subindo antes do outro.""",

            "front\\examples\\benchpress.mp4", "front\\examples\\benchpress_output.mp4" ]
        
    if "wrong" in label:
        data[0] += " (Execução com erros)"
    else: 
        data[0] += " (Execução correta)"
        
    return data

class TreineCertoApp(ctk.CTk):
    def __init__(self, temp_dir = None):
        
        super().__init__()
        self.title("Treine Certo")
        self.geometry("1920x1080")
        self.iconbitmap("front\\icon.ico")
        
        self.current_video_path = None
        
        # Criar um arquivo temporário para os  vídeos
        self.temp_dir = temp_dir

        self.frames = {}
        for FrameClass in (MainMenu, ProcessandoFrame, ResultadoFrame, InfoFrame, ExerciciosFrame, ExercicioDetalheFrame):
            frame = FrameClass(self)
            self.frames[FrameClass] = frame
            frame.place(relwidth = 1, relheight = 1)
            
        # Classe para o processamento do OpenPose
        self.pose_analyzer = PoseAnalyzer()
        
        # Carrega o Modelo da Rede
        self.classifier = TemporalGCNClassifier(in_channels = 50, hidden_channels = 128, out_channels = len(LABEL_MAP))
        model = torch.load("models/model_20250403_135511.pth", map_location = "cpu")
        self.classifier.model.load_state_dict(model)

        self.show_frame(MainMenu)
        
    def predict_label(self, json_path):

        label = self.classifier.predict(json_path)
        return label

    def show_frame(self, frame_class, **kwargs):
        frame = self.frames[frame_class]
        
        if hasattr(frame, "update_content"):
            frame.update_content(**kwargs)
            
        frame.tkraise()
        
    def processar_video(self):
        
        def task():
            output_path, json_path = self.pose_analyzer.process_video(video_path  = self.current_video_path,
                                                                      output_path = os.path.join(self.temp_dir, os.path.basename(self.current_video_path)),
                                                                      save_data   = True,
                                                                      replace     = True
                                                                    )
            
            print(f"Processado em: {json_path} e {output_path}")
    
            pred_label = self.predict_label(json_path)
        
            # Chama atualização da UI na thread principal
            self.after(0, lambda: self.show_frame(ResultadoFrame, 
                                                  video_path = output_path, 
                                                  json_path   = json_path , 
                                                  pred_label  = pred_label))
            
        threading.Thread(target=task).start()

btn_font = ("Segoe UI", 20, "bold")

class MainMenu(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        ctk.CTkLabel(self, text = "Treine Certo", font = ("Segoe UI", 80, "bold")).pack(pady = (200, 10))
        ctk.CTkLabel(self, text = "Análise de vídeos e identificação de exercícios", font = ("Segoe UI", 18)).pack(pady = (0,30))

        ctk.CTkButton(self, text = "Analisar Vídeo"        , font = btn_font, border_spacing = 10, width = 300, command = self.abrir_video).pack(pady = 20)
        ctk.CTkButton(self, text = "Exercícios Disponíveis", font = btn_font, border_spacing = 10, width = 300, command = lambda: master.show_frame(ExerciciosFrame)).pack(pady = 20)
        ctk.CTkButton(self, text = "Informações / Ajuda"   , font = btn_font, border_spacing = 10, width = 300, command = lambda: master.show_frame(InfoFrame      )).pack(pady = 20)

    def abrir_video(self):
        path = fd.askopenfilename(filetypes = [("MP4 Files", "*.mp4")])
        if path:
            self.master.current_video_path = path
            self.master.show_frame(ProcessandoFrame)

class ProcessandoFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        ctk.CTkLabel(self, text = "Processando vídeo"    , font = ("Segoe UI", 40, "bold")).pack(pady = (240, 15))
        ctk.CTkLabel(self, text = "Por favor, aguarde...", font = ("Segoe UI", 20, "bold")).pack()

        self.progress = ctk.CTkProgressBar(self, mode = "indeterminate")
        self.progress.pack(pady = (10, 0))
        self.progress.start()

    def update_content(self):
        # Espera 200 ms para dar tempo da interface redesenhar antes de processar
        self.after(200, lambda: threading.Thread(target = self.master.processar_video, daemon = True).start())

class ResultadoFrame(ctk.CTkFrame):
    
    def __init__(self, master):
        super().__init__(master)

        self.video_path = None
        self.json_path  = None
        
        self.slider_dragging = False
        self._pending_seek_value = 0

        # Classe estimada
        self.text_class = ctk.CTkLabel(self, text="Classe Estimada: Exemplo", font = ("Segoe UI", 40, "bold"))
        self.text_class.pack(pady = 10)

        # Container com tamanho fixo para o vídeo
        self.video_frame = ctk.CTkFrame(self, width = 960, height = 540, fg_color = "transparent")
        self.video_frame.pack(pady = 10)
        self.video_frame.pack_propagate(False)

        # Player dentro desse frame
        self.player = TkinterVideo(master = self.video_frame, scaled = True)
        self.player.pack(fill = "both", expand = True)

        # Slider de tempo
        self.slider = ctk.CTkSlider(self, from_ = 0, command = self.on_slider_move)
        self.slider.pack(fill = "x", padx = 40, pady = 10)

        self.slider.bind("<Button-1>"       , lambda e: self.set_slider_dragging(True))
        self.slider.bind("<ButtonRelease-1>", lambda e: self.set_slider_dragging(False))

        # Controles
        controls = ctk.CTkFrame(self, fg_color = "transparent")
        controls.pack(pady = 5)

        ctk.CTkButton(controls, font = btn_font, text = "▶ Play"     , command = self.play_video   ).pack(side = "left", padx = 10)
        ctk.CTkButton(controls, font = btn_font, text = "⏸ Pause"    , command = self.pause_video  ).pack(side = "left", padx = 10)
        ctk.CTkButton(controls, font = btn_font, text = "⏮ Reiniciar", command = self.restart_video).pack(side = "left", padx = 10)

        # Frame para textos lado a lado
        self.text_frame = ctk.CTkFrame(self, fg_color = "transparent")
        self.text_frame.pack(fill = "both", expand = True, padx = 20, pady = 10)

        # Textbox 1 - Execução Correta
        self.textbox_desc = ctk.CTkTextbox(self.text_frame, height = 200, wrap = "word", activate_scrollbars = True)
        self.textbox_desc.configure(font = ("Segoe UI", 16), state = "normal", padx = 20, pady = 15)
        self.textbox_desc.pack     (side = "left", fill = "both", expand = True, padx = (0, 10))

        # Textbox 2 - Erros Comuns
        self.textbox_erro = ctk.CTkTextbox(self.text_frame, height = 200, wrap = "word", activate_scrollbars = True)
        self.textbox_erro.configure(font = ("Segoe UI", 16), state = "normal", padx = 20, pady = 15)
        self.textbox_erro.pack     (side = "right", fill = "both", expand = True, padx = (10, 0))
        
        # Navegação
        ctk.CTkButton(self, text = "Retornar ao Menu", font = btn_font, border_spacing = 10, command = lambda: master.show_frame(MainMenu)).pack(side = "right", anchor = "se", padx = 10, pady = 10 )

        # Botões de salvar
        ctk.CTkButton(self, text = "Salvar Vídeo", font = btn_font, border_spacing = 10, command = self.salvar_video).pack(side = "right", anchor = "se", padx = 10, pady = 10 )
        ctk.CTkButton(self, text = "Salvar JSON" , font = btn_font, border_spacing = 10, command = self.salvar_json ).pack(side = "right", anchor = "se", padx = 10, pady = 10 )

    def update_content(self, video_path, json_path, pred_label):
        
        self.video_path = video_path
        self.json_path  = json_path

        # Lê resolução do JSON
        with open(json_path, "r", encoding = "utf-8") as f:
            json_data = json.load(f)
            resolution = json_data.get("resolution", {})
            
            width  = resolution.get("width" , 480)
            height = resolution.get("height", 864)

        # Altura fixa do frame, calcula largura proporcional
        target_height = 540
        aspect_ratio = width / height if height else 9 / 16
        target_width = int(target_height * aspect_ratio)

        # Atualiza player
        self.player.destroy()
        self.player = TkinterVideo(master = self.video_frame, scaled = True,
                                width = target_width, height = target_height)

        self.player.place(relx = 0.5, rely = 0, anchor = "n", width = target_width, height = target_height)

        self.player.load(self.video_path)
        self.player.play()

        self.player.bind("<<Ended>>", lambda e: self.restart_video())

        # Configura o slider com o tempo máximo correto após carregar
        self.after(500, self.configurar_slider_maximo)
        
        self.pred_label    = pred_label
        self.exercise_data = get_exercise_data(pred_label) 

        self.text_class.configure(text = f"Exercício Estimado: {self.exercise_data[0]}")
    
        self.textbox_desc.delete("0.0", "end")
        self.textbox_desc.insert("0.0", self.exercise_data[1])

        self.textbox_erro.delete("0.0", "end")
        self.textbox_erro.insert("0.0", self.exercise_data[2])
        
        self.update_slider_loop()
        
    def configurar_slider_maximo(self):
        duration = self.player.video_info()["duration"]
        if duration and duration > 0:
            self.slider.configure(from_ = 0, to = duration * 1000)
            
    def play_video(self):
        self.player.play()

    def pause_video(self):
        self.player.pause()

    def restart_video(self):
        self.player.seek(0)
        self.player.play()

    def salvar_video(self):
        
        path = fd.asksaveasfilename(defaultextension = ".mp4", filetypes=[("MP4 Files", "*.mp4")])
        
        if path and self.video_path:
            shutil.copy2(self.video_path, path)

    def salvar_json(self):
        
        path = fd.asksaveasfilename(defaultextension = ".json", filetypes = [("JSON Files", "*.json")])
        if path and self.json_path:
            shutil.copy2(self.json_path, path)

    def update_slider_loop(self):
        if not self.slider_dragging:
            current_time = self.player.current_duration()
            self.slider.set(current_time * 1000)
            
        self.after(200, self.update_slider_loop)

    def on_slider_move(self, value):
        self._pending_seek_value = value
        timestamp = int(value / 1000)
        self.player.seek(timestamp)

        # Força atualização visual do frame atual, mesmo em pausa
        if self.player.is_paused():
            self.player.play()
            self.after(100, self.player.pause)

        self.slider.set(value)

    def set_slider_dragging(self, dragging):
        self.slider_dragging = dragging
        
        if not dragging:
            timestamp = int(self._pending_seek_value / 1000)
            self.player.seek(timestamp)

class ExerciciosFrame(ctk.CTkFrame):
    
    def __init__(self, master):
        super().__init__(master)

        ctk.CTkLabel(self, font = ("Segoe UI", 50, "bold"),
                     text = "Exercícios Disponíveis").pack(padx = 10, pady = (150, 10))
        
        ctk.CTkLabel(self, text = "Selecione um exercício para ver mais informações sobre sua execução, recomendações técnicas, possíveis erros e vídeos de exemplo.",
                     font = ("Segoe UI", 20), width = 800, anchor = "w", wraplength = 800, justify = "left").pack(padx = 10, pady = 25)

        squat_data      = get_exercise_data("Squat"     )
        deadlift_data   = get_exercise_data("Deadlift"  ) 
        benchpress_data = get_exercise_data("Benchpress")
        
        self.exercicios = [squat_data, deadlift_data, benchpress_data]

        for nome, desc, desc_erro, v1, v2 in self.exercicios:
            nome_limpo = nome.replace(" (Execução correta)", "")
            ctk.CTkButton(self, text = nome_limpo, font = btn_font, anchor = "w", width = 500, corner_radius = 6, border_spacing = 10, command = lambda n = nome_limpo, d = desc, d2 = desc_erro,
                          v1 = v1, v2 = v2: master.show_frame(ExercicioDetalheFrame, nome = n, desc = d, desc_erro = d2, video1 = v1, video2 = v2)).pack(padx = 20, pady = 20)

        ctk.CTkButton(self, text = "Voltar", font = btn_font, border_spacing = 10, command = lambda: master.show_frame(MainMenu)).pack(side = "right", anchor = "se", padx = 50, pady = 10)

class ExercicioDetalheFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.master = master
        self.video_path1 = None
        self.video_path2 = None
        
        self.slider_dragging = False
        self._pending_seek_value = 0
        
        self.label_nome = ctk.CTkLabel(self, text = "", font = ("Segoe UI", 40, "bold"))
        self.label_nome.pack(pady = 30)
        
        # Frame principal dos vídeos
        self.video_duo_frame = ctk.CTkFrame(self, fg_color = "transparent")
        self.video_duo_frame.pack(pady = 30, padx = 40)

        # Subframes dos vídeos
        self.video_left  = ctk.CTkFrame(self.video_duo_frame, width = 800, height = 450, fg_color = "transparent")
        self.video_right = ctk.CTkFrame(self.video_duo_frame, width = 800, height = 450, fg_color = "transparent")
        
        self.video_left .grid(row = 1, column = 0)
        self.video_right.grid(row = 1, column = 1)
        
        self.video_left.grid_propagate(False)
        self.video_right.grid_propagate(False)

        # Permitir que os vídeos se expandam dentro dos subframes
        self.video_left.rowconfigure    (1, weight =1 )
        self.video_left.columnconfigure (0, weight = 1)
        
        self.video_right.rowconfigure   (1, weight = 1)
        self.video_right.columnconfigure(0, weight = 1)

        # Label e player do vídeo 1 (esquerdo)
        ctk.CTkLabel(self.video_left, text = "Exemplo de Execução", font = ("Segoe UI", 20)).grid(row = 0, column = 0, pady = (0, 5))
        self.player1 = TkinterVideo(master = self.video_left, scaled = True)
        self.player1.grid(row = 1, column = 0, sticky = "nsew", padx = (0, 20))

        # Label e player do vídeo 2 (direito)
        ctk.CTkLabel(self.video_right, text = "Vídeo Processado", font = ("Segoe UI", 20)).grid(row = 0, column = 0, pady = (0, 5))
        self.player2 = TkinterVideo(master = self.video_right, scaled = True)
        self.player2.grid(row = 1, column = 0, sticky = "nsew", padx = (20, 0))

        # Controles
        controls = ctk.CTkFrame(self, fg_color = "transparent")
        controls.pack(pady=5)

        ctk.CTkButton(controls, font = btn_font, text = "▶ Play"     , command = self.play_video   ).pack(side = "left", padx = 10)
        ctk.CTkButton(controls, font = btn_font, text = "⏸ Pause"    , command = self.pause_video  ).pack(side = "left", padx = 10)
        ctk.CTkButton(controls, font = btn_font, text = "⏮ Reiniciar", command = self.restart_video).pack(side = "left", padx = 10)

        self.slider = ctk.CTkSlider(self, from_ = 0, to = 1000, state = "disabled")
        self.slider.pack(fill = "x", padx = 40, pady = 10)

        # Frame para textos lado a lado
        self.text_frame = ctk.CTkFrame(self)
        self.text_frame.pack(fill = "both", expand = True, padx = 20, pady = 10)

        # Textbox 1 - Execução Correta
        self.textbox_desc = ctk.CTkTextbox(self.text_frame, height = 200, wrap = "word", activate_scrollbars = True,)
        self.textbox_desc.configure(font = ("Segoe UI", 16), state = "normal", padx = 20, pady = 15)
        self.textbox_desc.pack     (side = "left", fill = "both", expand = True, padx = (0, 10))

        # Textbox 2 - Erros Comuns
        self.textbox_erro = ctk.CTkTextbox(self.text_frame, height = 200, wrap = "word", activate_scrollbars = True)
        self.textbox_erro.configure(font = ("Segoe UI", 16), state = "normal", padx = 20, pady = 15)
        self.textbox_erro.pack     (side = "right", fill = "both", expand = True, padx = (10, 0))
    
        ctk.CTkButton(self, text = "Voltar", font = btn_font, border_spacing = 10, command = self.voltar).pack(side = "right", anchor = "se", padx = 50, pady = 10)

    def update_content(self, nome, desc, desc_erro, video1, video2):
        self.label_nome.configure(text = nome)

        self.textbox_desc.delete("0.0", "end")
        self.textbox_desc.insert("0.0", desc)

        self.textbox_erro.delete("0.0", "end")
        self.textbox_erro.insert("0.0", desc_erro)
        
        self.update_slider_loop()
        
        if self.video_path1 == video1 and self.video_path2 == video2:
            return

        self.video_path1 = video1
        self.video_path2 = video2
        
        # Remove os players antigos (se existirem)
        self.player1.destroy()
        self.player2.destroy()

        # Cria novos players
        self.player1 = TkinterVideo(master = self.video_left, scaled = True)
        self.player1.grid(row = 1, column = 0, sticky="nsew", padx = (0, 20))

        self.player2 = TkinterVideo(master = self.video_right, scaled = True)
        self.player2.grid(row = 1, column = 0, sticky="nsew", padx = (20, 0))

        self.player1.load(video1)
        self.player2.load(video2)

        self.player1.play()
        self.player2.play()
        
        # Vincula evento de fim para reiniciar o player automaticamente
        self.player1.bind("<<Ended>>", lambda e: self.restart_video())

    def play_video(self):
        self.player1.play()
        self.player2.play()

    def pause_video(self):
        self.player1.pause()
        self.player2.pause()

    def restart_video(self):
        self.player1.seek(0)
        self.player2.seek(0)
        self.play_video()
        
    def update_slider_loop(self):
        if not self.slider_dragging:
            current_time = self.player1.current_duration()
            duration = self.player1.video_info()["duration"]
            if duration > 0:
                slider_value = (current_time / duration) * 1000
                self.slider.set(slider_value)
                
        self.after(200, self.update_slider_loop)

    def voltar(self):
        self.pause_video()
        self.master.show_frame(ExerciciosFrame)

class InfoFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # Frame externo com canvas e scrollbar
        scrollable_frame = ctk.CTkScrollableFrame(self, label_text = "", fg_color = "transparent")
        scrollable_frame.pack(fill = "both", expand = True, padx = (40), pady = 0)

        ctk.CTkLabel(scrollable_frame, font = ("Segoe UI", 50, "bold"), text = "Informações / Ajuda").pack(pady = 20)

        # Subtítulo 1
        ctk.CTkLabel(scrollable_frame, text = "Sobre o programa", font = ("Segoe UI", 26, "bold")).pack(pady = (0, 0), padx = 40, anchor = "w")

        texto_geral = """O Treine Certo é uma ferramenta desenvolvida para auxiliar usuários na análise da execução de exercícios físicos a partir de vídeos. O sistema combina técnicas de visão computacional e inteligência artificial para identificar, classificar e avaliar a execução de movimentos, utilizando um modelo treinado com diversos vídeos de exemplo.

Ao enviar um vídeo, o sistema processa a movimentação corporal com base em 25 pontos-chave extraídos pelo OpenPose, uma tecnologia de estimativa de pose que identifica articulações como joelhos, ombros, quadris e tornozelos. As coordenadas X e Y de cada ponto são analisadas por uma Rede Neural Convolucional de Grafos (GCN) temporal, que interpreta o padrão de movimento e classifica o exercício executado, distinguindo entre execuções corretas e incorretas.

Fatores como a posição do praticante no quadro e a duração do vídeo são automaticamente normalizados pelo sistema. No entanto, para garantir uma análise precisa, é fundamental que o vídeo seja gravado com um bom posicionamento e ângulo de câmera, adequados ao tipo de exercício e que apenas o praticante esteja presente no vídeo, evitando que a estimativa de pose mude de alvo ao decorrer do vídeo. Como a análise depende da detecção precisa dos pontos corporais, recomenda-se seguir as orientações fornecidas para obter resultados confiáveis."""

        box1 = ctk.CTkTextbox(scrollable_frame, wrap = "word", height = 250)
        box1.configure(font = ("Segoe UI", 16), padx = 20, pady = 15)
        box1.insert("0.0", texto_geral)
        box1.pack(pady = (0, 30), padx = 60, fill = "x", expand = False)

        # Subtítulo 2
        ctk.CTkLabel(scrollable_frame, text = "Orientações sobre a captura de vídeo", font = ("Segoe UI", 26, "bold")).pack(pady = (0, 0), padx = 40, anchor = "w")

        texto_posicionamento = """Para garantir uma análise mais precisa, é essencial que o vídeo seja gravado com um bom posicionamento e ângulo de câmera, adequados ao exercício executado. Como o sistema depende da detecção precisa dos pontos corporais, é fundamental seguir as seguintes recomendações:

- Evite ângulos que ocultem partes do corpo, como braços ou pernas atrás de equipamentos ou fora do campo de visão. Embora o OpenPose tente estimar a posição dos membros ocultos, isso pode gerar erros de detecção. (Atenção especial à gravação de vídeos de Supino com Barra).
- Posicione a câmera a aproximadamente 45 graus à frente da pessoa que está executando o exercício. Esse ângulo geralmente oferece uma visão equilibrada da movimentação de braços, joelhos e coluna.
- Centralize o praticante no quadro, mantendo o quadril próximo ao centro da imagem. Evite que a pessoa fique muito próxima ou distante da câmera, pois isso melhora a detecção dos pontos-chave e a qualidade da análise do movimento.
- Considere o objetivo da análise para escolher o melhor ângulo. Por exemplo, para avaliar um agachamento com barra e observar possíveis desvios de joelhos (como o valgismo), uma perspectiva frontal direta é mais adequada, pois permite visualizar claramente o alinhamento entre quadris, joelhos e tornozelos.
- No caso do Supino com Barra, recomenda-se uma posição lateral a 45 graus e levemente elevada em relação ao praticante, minimizando a oclusão dos braços e ombros.

A adoção dessas boas práticas de gravação aumenta significativamente a precisão da análise e reduz o risco de classificações incorretas."""

        box2 = ctk.CTkTextbox(scrollable_frame, wrap = "word", height = 330)
        box2.configure(font = ("Segoe UI", 16), padx = 20, pady = 15)
        box2.insert("0.0", texto_posicionamento)
        box2.pack(pady = (0, 20), padx = 60, fill = "x", expand = False)

        # Linha de imagens
        img_row = ctk.CTkFrame(scrollable_frame, fg_color = "transparent")
        img_row.pack(pady = (0, 20), padx = 60, fill = "x")

        img_size = (330, 220)  # novo tamanho maior para usar o espaço disponível

        img_paths = [
            ("front\\angles\\45 left.png" , "Frontal em 45º (esquerda)"),
            ("front\\angles\\frontal.png" , "Frontal direta"),
            ("front\\angles\\45 right.png", "Frontal em 45º (direita)")
        ]

        for path, label_text in img_paths:
            col = ctk.CTkFrame(img_row, fg_color = "transparent")
            col.pack(side = "left", expand = True, padx = 10)

            img = self.load_img(path, img_size)
            ctk.CTkLabel(col, image = img, text = "").pack(pady = (0, 5))
            ctk.CTkLabel(col, text = label_text, font = ("Segoe UI", 14, "bold")).pack()

        # Botão voltar
        ctk.CTkButton(scrollable_frame, text = "Voltar", font = btn_font, border_spacing = 10,
                      command = lambda: master.show_frame(MainMenu)).pack(side = "right", anchor = "se", padx = 50, pady = 10)


    def load_img(self, path, size):
        img = Image.open(path)
        img = img.resize(size)
        
        return ctk.CTkImage(light_image = img, dark_image = img, size = size)

if __name__  ==  "__main__":
    
    temp_dir = tempfile.mkdtemp()
    app = TreineCertoApp(temp_dir)
    app.mainloop()

