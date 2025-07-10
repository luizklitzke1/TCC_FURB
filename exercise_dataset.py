import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import numpy as np

import os
import json
import pickle

from pose_analyzer import BODY_25_POINTS, B25_R

# Mapeamento das labels (ajuste conforme suas classes reais)
LABEL_MAP = {
    "Deadlift"          : 0,
    "Deadlift_wrong"    : 1,
    "Squat"             : 2,
    "Squat_wrong"       : 3,
    "Benchpress"        : 4,
    "Benchpress_wrong"  : 5
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Converte uma sequência de keypoints (tensor com shape (target_len, 25, 2)) em um objeto Data do PyTorch Geometric.
def create_data_object(sequence, label = None):
    
    target_len = sequence.shape[0]
    
    # Achata cada frame (cada frame terá 50 features)
    frame_features = sequence.view(target_len, -1)  # shape: (target_len, 50)

    # Cria edge_index conectando o frame i ao i+1 (bidirecional)
    if target_len > 1:
        indices = [[i, i + 1] for i in range(target_len - 1)]
        edge_index = torch.tensor(indices, dtype = torch.long).t().contiguous()
        
        # Arestas reversas
        rev_edge_index = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, rev_edge_index], dim = 1)
        
    else:
        edge_index = torch.empty((2, 0), dtype = torch.long)

    # Converte a label (string) para inteiro
    if (label is not None) and (label in LABEL_MAP):
        y = torch.tensor(LABEL_MAP[label], dtype = torch.long)
    else:
        y = torch.tensor(-1, dtype = torch.long)
    
    # Cria e retorna o objeto Data
    return Data(x = frame_features, edge_index = edge_index, y = y)

# Método para reduzir a taxa de frames
def downsample_frames(frames, original_fps, target_fps):
   
    if original_fps <= target_fps:
        return frames
    
    # Calcula o número de frames após downsampling
    new_length = int(round(len(frames) * target_fps / original_fps))
    
    # Gera os índices dos frames que serão mantidos
    indices = np.linspace(0, len(frames) - 1, num = new_length, dtype = int)
    
    return [frames[i] for i in indices]

def normalize_keypoints(sequence): # sequence: array de shape (frames, 25, 2)
    
    # Utiliza o midhip de cada frame como ponto de referência
    #Obs: Tentei treinamentos com o MidHip do primeiro frame como ponto de referência,
    #para pegar o movimento dele ao longo do vídeo mas não funcionou bem.
    midhip = sequence[:, B25_R["MidHip"], :]
    
    # Centraliza todos os pontos em relação ao MidHip
    sequence_centered = sequence - midhip[:, np.newaxis, :] 
   
    # Calcula a maior distância em cada frame para normalizar a escala - Esse cara ajuda a manter a escala do corpo de uma pessoa para outra.
    max_dist = np.max(np.linalg.norm(sequence_centered, axis = 2), axis = 1, keepdims = True)
    normalized_sequence = sequence_centered / (max_dist[:, np.newaxis, :] + 1e-8)

    return normalized_sequence

# Função de interpolação temporal - para ajustar o tamanho dos vídeos
def temporal_interpolation(sequence, target_len):
    
    original_len = len(sequence)
    
    x_original = np.linspace(0, 1, original_len)
    x_target   = np.linspace(0, 1, target_len)

    # Interpolação linear para cada ponto do corpo
    interpolated_sequence = np.zeros((target_len, sequence.shape[1], sequence.shape[2]))
    
    for joint in range(sequence.shape[1]):  # Para cada ponto do corpo
        for coord in range(sequence.shape[2]):  # Para x e y
            interpolated_sequence[:, joint, coord] = np.interp(x_target, x_original, sequence[:, joint, coord])
    
    return interpolated_sequence

# Processamento de JSON de um vídeo para item normalizado
def process_json_file(filepath, target_len, target_fps, parts = BODY_25_POINTS):
    
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    original_fps = json_data["fps"   ]
    data_frames  = json_data["frames"]
    
    # Reduz a taxa de frames para target_fps
    data_frames = downsample_frames(data_frames, original_fps, target_fps)
    
    sequence = []
    for frame_data in data_frames:
        keypoints = frame_data['keypoints']
        frame = []
        # Para cada ponto, extrai x e y a partir da lista; assume que o formato é [x, y, confidence]
        for point_name in parts.values():
            
            # Usa keypoints.get() para obter a lista; se não existir, usa [0,0,0]
            point = keypoints.get(point_name, [0, 0, 0])
            frame.append([point[0], point[1]])
            
        sequence.append(frame)
        
    sequence = np.array(sequence)  # shape: (frames, 25, 2)
    sequence = normalize_keypoints(sequence)
    sequence = temporal_interpolation(sequence, target_len)
    
    return sequence  # (target_len, 25, 2)

# Junta todos os métodos de processamento e conversão de dados em uma só chamada
def json_to_data(filepath, label = None, target_len = 15, target_fps = 15):
    
    target_len_frames = target_len * target_fps
    
    sequence = process_json_file(filepath, target_len = target_len_frames, target_fps = target_fps)
    sequence = torch.tensor(sequence, dtype = torch.float32)
    data     = create_data_object(sequence, label)
    
    return data

class ExerciseDataset(Dataset):
    def __init__(self, root_dir, target_len = 15, target_fps = 15, cache = True, aug = True):
        
        self.root_dir   = root_dir
        self.target_len = target_len
        self.target_fps = target_fps
        
        # Nome do arquivo de cache
        self.cache_file = os.path.join(root_dir, f"cache_{target_len}_{target_fps}_{aug}.pkl")
        print("Arquivo de cache para o dataset: ", self.cache_file)
            
        self.samples   = [] # Lista de tuplas pré processamento para ref (filepath, label)
        self.data_list = [] # Lista para armazenar os dados processados

        # Se o cache estiver habilitado e existir, carregue os dados pré-processados
        if cache and os.path.exists(self.cache_file):
            print("Carregando cache de samples e dados processados...")
            
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.samples = cache_data["samples"]
                self.data_list = cache_data["data_list"]
                
            print("Cache carregado.")
            
        else:
            print("Pré-processando dados... (isso pode demorar)")
            
            # Constrói a lista de amostras apenas se não houver cache
            self.samples = []
            
            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for filename in os.listdir(label_dir):
                    if filename.endswith('.json') and (aug == True or not("aug" in filename)):
                        self.samples.append((os.path.join(label_dir, filename), label))
            
            self.data_list = []
            total = len(self.samples)
            
            for idx, (filepath, label) in enumerate(self.samples):
                
                data = json_to_data(filepath, label, target_len = self.target_len, target_fps = self.target_fps)
                
                self.data_list.append(data)
                
                # Exibe o progresso (sobrescreve a mesma linha)
                print(f"Processando amostra {idx + 1} / {total} ({((idx+1)/total)*100:.2f}%)", end = "\r")
                
            print("\nPré-processamento concluído.")
            
            if cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({"samples": self.samples, "data_list": self.data_list}, f)
                    
                print("Dados salvos no cache.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_item_path(self, idx):
        filepath, _ = self.samples[idx]
        return filepath
    