import os
from pose_analyzer import PoseAnalyzer
import sys
import tempfile
import cv2

# Classe utilitária para processar todos os vídeos de exercícios à partir de um diretório de entrada
class VideoProcessor:

    def __init__(self, exercise, input_dir, output_dir, selected_points = None, replace = False, force_vertical = False):
        self.exercise = exercise
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.selected_points = selected_points
        self.replace = replace
        self.force_vertical = force_vertical # Alguns vídeos, por algumo motivo, estão na vertical apenas por metadados, mas o vídeo em si está na horizontal.
        self.pose_analyzer = PoseAnalyzer(selected_points = self.selected_points)
    
    def process_all_videos(self):
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Diretório de entrada '{ self.input_dir }' não encontrado")
        
        # Cria o diretório de saída se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Percorre todas as subpastas (tipos de exercícios)
        for exercise_type in os.listdir(self.input_dir):
            
            if self.exercise != "all" and self.exercise.lower() != exercise_type.lower():
                continue
            
            exercise_path = os.path.join(self.input_dir, exercise_type)
            if not os.path.isdir(exercise_path):
                continue

            print(f"\nProcessando tipo de exercício: { exercise_type }")
            
            # Cria pasta correspondente na saída
            output_exercise_path = os.path.join(self.output_dir, exercise_type)
            if not os.path.exists(output_exercise_path):
                os.makedirs(output_exercise_path)
            
            self.process_exercise_folder(exercise_path, output_exercise_path)
    
    def process_exercise_folder(self, exercise_path, output_path):
        
        for source_name in os.listdir(exercise_path):
            source_path = os.path.join(exercise_path, source_name)
            
            if not os.path.isdir(source_path):
                continue
            
            print(f"    Processando fonte: { source_name }")
            
            for file_name in os.listdir(source_path):
                if os.path.splitext(file_name.lower())[1] != ".mp4":
                    continue
                
                video_path = os.path.join(source_path, file_name)
                output_file_name = f"{ source_name }_{ file_name }"
                output_video_path = os.path.join(output_path, output_file_name)

                print(f"\n        Processando vídeo: { file_name }")
                self.process_video(video_path, output_video_path)

    def process_force_vertical(self, caminho_video):
        
        # Criar um arquivo temporário para o novo vídeo
        temp_dir = tempfile.mkdtemp()
        caminho_temporario = os.path.join(temp_dir, 'video_temp.mp4')
        
        # Abrir o vídeo de entrada
        cap = cv2.VideoCapture(caminho_video)
        
        # Obter propriedades do vídeo
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
    
        # Verificar se a largura é maior que a altura (indicando rotação incorreta)
        if width <= height:
             # Limpar recursos temporários se nenhuma correção for necessária
            cap.release()
            os.rmdir(temp_dir)
            print("            Vídeo corretamente na vertical! Nenhuma ação necessária.")   
            return
        
        print("            Ajustando vídeo para vertical.")   
        
        # Definir codec de vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Criar objeto VideoWriter com dimensões corretas
        out = cv2.VideoWriter(caminho_temporario, fourcc, fps, (height, width))
        
        # Processar cada quadro
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotacionar o quadro 90 graus no sentido horário
            frame_rotacionado = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Gravar o quadro rotacionado
            out.write(frame_rotacionado)
        
        # Liberar recursos
        cap.release()
        out.release()
        
        # Substituir o arquivo original
        try:
            # Remover o arquivo original
            os.remove(caminho_video)
            
            # Renomear o arquivo temporário para o nome original
            os.rename(caminho_temporario, caminho_video)
            print("            Vídeo corrigido e substituído no local original")
            
        except Exception as e:
            print(f"            Erro ao substituir o arquivo: {e}")
            print(f"            Vídeo temporário salvo em: {caminho_temporario}")
            
    def process_video(self, video_path, output_video_path):
        # Verifica se o vídeo já foi processado e se a opção de substituição não está ativada
        if os.path.exists(output_video_path) and not self.replace:
            print(f"        Vídeo já processado: { output_video_path }")
            return
        
        if (self.force_vertical):
            self.process_force_vertical(video_path)
            
        try:
            self.pose_analyzer.process_video(
                video_path  = video_path,
                output_path = output_video_path,
                save_data   = True
            )
            print(f"        Vídeo processado e salvo em: {output_video_path}")
            
        except Exception as e:
            print(f"        Erro ao processar vídeo {video_path}: {str(e)}")
    
    def run(self):
        print(f"Iniciando processamento de vídeos de exercícios...")
        print(f"Pasta de input: {self.input_dir}")
        print(f"Pasta de output: {self.output_dir}")
        try:
            self.process_all_videos()
            print("Processamento concluído com sucesso!")
        except Exception as e:
            print(f"Erro durante o processamento: {str(e)}")

if __name__ == "__main__":
    
    # Processar argumentos
    n = len(sys.argv)
    if n < 2:
        print ("Informe o nome do exercício a ser processado via linha de comando.\n")
        print ("Ou informe 'all' para todos.\n")
        sys.exit(1)
        
    exercise = sys.argv[1]
    print ("Exercício: ", exercise)
    
    force_vertical = False
    if (n > 2):
        force_vertical = sys.argv[2] == "True"
        
    print ("Forçar vertical: ", force_vertical)
    
    externo = False
    if exercise == "externo":
        externo = True
        exercise = "all"
        
        input_path = "input_externo"
        output_path = "output_externo"
    else:
        input_path = "input"
        output_path = "output"
        
    processor = VideoProcessor(exercise, input_path, output_path, replace = False, force_vertical = force_vertical)
    processor.run()
