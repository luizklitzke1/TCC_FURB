import cv2
import json
import os
import sys
from sys import platform

try:
    # Import Openpose
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Windows Import
    if platform == "win32":
        sys.path.append(os.path.join(dir_path, "openpose_build/python/openpose/Release"))
        os.environ["PATH"]  = os.environ["PATH"] + ";" + os.path.join(dir_path, "openpose_build/x64/Release") + ";" + os.path.join(dir_path, "openpose_build/bin")
        import pyopenpose as op
    else:
        raise Exception("Plataforma não suportada.")
    
except ImportError as e:
    print("Error: Falha ao improtar a library do OpenPose. Lembrar de habilitar `BUILD_PYTHON` no CMake e copiar os arquivos de Python para o path correto.")
    raise e
except Exception as e:
    print(e)
    raise e

# Lista de pontos do modelo BODY_25
BODY_25_POINTS = {
    0 : "Nose"     , 1 : "Neck"  , 2 : "RShoulder", 3 : "RElbow"   , 4 : "RWrist" ,
    5 : "LShoulder", 6 : "LElbow", 7 : "LWrist"   , 8 : "MidHip"   , 9 : "RHip"   ,
    10: "RKnee"    , 11: "RAnkle", 12: "LHip"     , 13: "LKnee"    , 14: "LAnkle" ,
    15: "REye"     , 16: "LEye"  , 17: "REar"     , 18: "LEar"     , 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel" , 22: "RBigToe"  , 23: "RSmallToe", 24: "RHeel"  
}

# Dicionário reverso, para facilitar pegar o índice de um ponto pelo nome depois
B25_R = {v: k for k, v in BODY_25_POINTS.items()}

class PoseAnalyzer:
    
    def __init__(self, selected_points = None):
        if selected_points is None:
            selected_points = list(BODY_25_POINTS.keys())
            
        self.selected_points = selected_points

        # Parametros customizados (ref em openpose/flags.hpp)  
        # ou https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp)
        params = {
            "model_folder"     : "openpose_build/models/",
            "model_pose"       : "BODY_25"               ,  # Usando BODY_25 para maior precisão
            "net_resolution"   : "-1x368"                ,  # Width negativa para manter o aspect ratio - Considerar casos de vídeos que tem que ser forçados para vertical
            "number_people_max": 1                       ,  # Assumir que temos uma pessoa por vídeo
            "keypoint_scale"   : 3                       ,  # 3 = [-1,1], 0 = [0,width], 1 = [0,1]
            "disable_blending" : False                   ,
        }

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        # Inicializar o datum object
        self.datum = op.Datum()

    def process_video(self, video_path, output_path, save_data = True, replace = False):
        
        #Abrir o video
        cap = cv2.VideoCapture(video_path)
        
        fps          = cap.get(cv2.CAP_PROP_FPS)
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT ))

        # Garantir que o diretório do output_path existe
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if replace == False and os.path.exists(output_path):
            print(f"Arquivo {output_path} já existe. Pulando processamento.")
            return

        # Inicializar o video writer para gravar o video processado
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height)
        )

        # Inicializar lista para armazenar os dados de posição
        pose_data = []
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Processar um frame
            keypoints, annotated_frame = self.process_frame(frame)
            
            # Salvar o frame processado
            out.write(annotated_frame)

            # Salvar os keypoints se existirem
            if save_data and keypoints is not None:

                # Inicializar dicionário vazio para armazenar os keypoints selecionados
                selected_keypoints = {}

                for idx in self.selected_points:
                    if idx < len(keypoints):
                        
                        # Obter o nome do ponto (ex: "Nose", "Neck", etc)
                        point_name        = BODY_25_POINTS[idx]
                        point_coordinates = keypoints[idx].tolist()
                        
                        # Adicionar ao dicionário
                        selected_keypoints[point_name] = point_coordinates

                frame_data = { "frame": frame_number, "keypoints": selected_keypoints }
                pose_data.append(frame_data)

            frame_number += 1
            
        # Release na captura e no video writer
        cap.release()
        out.release()

        # Salvar pontos em um arquivo JSON
        if save_data:
            
            resolution = { "width": frame_width, "height": frame_height }
            
            full_data = {
                "fps"         : fps,
                "frame_count" : frame_count,
                "resolution"  : resolution,
                "frames"      : pose_data
            }
        
            json_path = output_path.rsplit(".", 1)[0] + "_data.json"
            
            with open(json_path, "w") as f:
                json.dump(full_data, f)

            return output_path, json_path
        
        return output_path, None

    def process_frame(self, frame):
        
        # Preparar o frame para processamento
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))

        keypoints = self.datum.poseKeypoints
        if keypoints is None:
            return None, frame

        # Utilizar o objeto interno do OpenPose para externar os keypoints no frame
        annotated_frame = self.datum.cvOutputData

        return keypoints[0], annotated_frame
