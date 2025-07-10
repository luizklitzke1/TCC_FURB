import os
import json
import math
import random
import sys

# Dicionário para trocar labels de lado no flip horizontal
FLIP_LABELS = {
    "RShoulder": "LShoulder",
    "RElbow"   : "LElbow",
    "RWrist"   : "LWrist",
    "RHip"     : "LHip",
    "RKnee"    : "LKnee",
    "RAnkle"   : "LAnkle",
    "REye"     : "LEye",
    "REar"     : "LEar",
    "RBigToe"  : "LBigToe",
    "RSmallToe": "LSmallToe",
    "RHeel"    : "LHeel"
}

# Inverso
FLIP_LABELS.update({v: k for k, v in FLIP_LABELS.items()})

# Flip horizontal em x \in [0,1] e troca as labels de lado, ex: RShoulder -> LShoulder.
def flip_horizontal(frame_keypoints):
    new_keypoints = { }
    
    for label, coords in frame_keypoints.items():
        
        # Se o label tiver no dicionário de flip, troca
        flipped_label = FLIP_LABELS.get(label, label)
        x, y, c = coords
        x_new = 1 - x  # flip
        
        new_keypoints[flipped_label] = [x_new, y, c]
        
    return new_keypoints

# Acha o MidHip no frame para ponto de ref. Se não existir, retorna None ou [0,0,0].
def find_midhip(frame_keypoints):
    
    if "MidHip" in frame_keypoints:
        return frame_keypoints["MidHip"]
    
    # Se não existir nenhum dos dois, retorna [0,0,0] ou None
    return [0, 0, 0]

#Aplica um zoom em torno do MidHip. 
def scale_frame(frame_keypoints, alpha = 1.1):
   
    midhip = find_midhip(frame_keypoints)
    
    x_m, y_m, conf = midhip
    new_keypoints = { }
    
    for label, coords in frame_keypoints.items():
        x, y, c = coords
        
        # Subtrair midhip
        x_shift = x - x_m
        y_shift = y - y_m
        
        # Aplicar escala
        x_shift *= alpha
        y_shift *= alpha
        
        # Reposicionar
        x_new = x_shift + x_m
        y_new = y_shift + y_m
        
        new_keypoints[label] = [x_new, y_new, c]
        
    return new_keypoints

# Rotaciona em torno do MidHip por 'degrees'.
def rotate_frame(frame_keypoints, degrees = 5):

    rad = math.radians(degrees)
    cos_ = math.cos(rad)
    sin_ = math.sin(rad)
    midhip = find_midhip(frame_keypoints)
    x_m, y_m, _ = midhip
    new_keypoints = {}
    
    for label, coords in frame_keypoints.items():
        
        x, y, c = coords
        
        # Subtrair midhip
        x_shift = x - x_m
        y_shift = y - y_m
        
        # Rotação 2D
        x_rot = x_shift*cos_ - y_shift*sin_
        y_rot = x_shift*sin_ + y_shift*cos_
        
        # Reposicionar
        x_new = x_rot + x_m
        y_new = y_rot + y_m
        new_keypoints[label] = [x_new, y_new, c]
        
    return new_keypoints

# Adiciona ruído gaussiano de amplitude noise_level em x e y.
def add_noise(frame_keypoints, noise_level = 0.01):
   
    new_keypoints = {}
    
    noise = random.uniform(-noise_level, noise_level)
    
    for label, coords in frame_keypoints.items():
        
        x, y, c = coords
        
        x_new = x + noise
        y_new = y + noise
        
        new_keypoints[label] = [x_new, y_new, c]
        
    return new_keypoints


# Aplica as transformações no json_data (já carregado).
def augment_json(json_data, do_flip = False, scale_alpha = None, rotate_deg = None, noise = None):
    
    # Copia o dicionário original
    aug_data = {
        "fps"        : json_data["fps"        ],
        "frame_count": json_data["frame_count"], 
        "resolution" : json_data["resolution" ],
        "frames"     : []
    }
    
    for frame_info in json_data["frames"]:
        
        frame_id  = frame_info["frame"    ]
        keypoints = frame_info["keypoints"]
        
        # Aplica transformações
        if do_flip:
            keypoints = flip_horizontal(keypoints)
            
        if scale_alpha is not None:
            keypoints = scale_frame(keypoints, alpha = scale_alpha)
            
        if rotate_deg is not None:
            keypoints = rotate_frame(keypoints, degrees = rotate_deg)
            
        if noise is not None:
            keypoints = add_noise(keypoints, noise_level = noise)
            
        # Monta novo frame
        new_frame = {
            "frame"    : frame_id,
            "keypoints": keypoints
        }
        
        aug_data["frames"].append(new_frame)
    
    # Ajustar frame_count se necessário
    aug_data["frame_count"] = len(aug_data["frames"])
    
    return aug_data

# Gera múltiplas versões do JSON com as transformações especificadas e salva em output_dir.
def generate_augmented_versions(input_json, output_dir, flips = [False, True], scales = [1.1, 0.9], rotations = [5, -5], noises = [None, 0.01]):
   
    if "aug" in input_json:
        return # Não processa se já for um JSON aumentado
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base         = os.path.basename(input_json) # Extrai o nome do arquivo com extensão
    base_name, _ = os.path.splitext(base)       # Separa o nome da extensão

    with open(input_json, 'r') as f:
        data = json.load(f)

    i = 0
    
    # Para cada combinação de transformações, gera um novo JSON
    
    for flip in flips:
        for sc in scales:
            for rot in rotations:
                for nz in noises:
                    
                    # Pulamos a combinação que não muda nada :D
                    if (not flip) and (sc == 1.0) and (rot == 0) and (nz is None):
                        continue
                    
                    aug = augment_json(data, do_flip = flip, scale_alpha = sc if sc != 1.0 else None,
                                       rotate_deg =rot if rot != 0 else None,
                                       noise = nz)
                    
                    # Nome do novo JSON com as transformações aplicadas
                    transform_name = f"flip_{flip}_scale_{sc}_rot_{rot}_noise_{nz}"
                    out_name       = f"aug_{base_name}_{i}_{transform_name}.json"
                    out_path       = os.path.join(output_dir, out_name)
                    
                    with open(out_path, 'w') as f:
                        json.dump(aug, f, indent = 4)

                    i += 1

if __name__ == "__main__":
    
    # Processar argumentos
    n = len(sys.argv)
    if n < 2:
        print ("Informe o nome do exercício a ser processado via linha de comando.\n")
        print ("Ou informe 'all' para todos.\n")
        sys.exit(1)
        
    exercise = sys.argv[1]
    print ("exercício: ", exercise)
    
    for label in os.listdir("output"):
        
        if exercise != "all" and exercise.lower() != label.lower():
                continue
        
        label_dir = os.path.join("output", label)
        
        print(f"\nGerando versões aumentadas para a label: {label}\n")
        
        files = os.listdir(label_dir)
        total = len(files) / 2
        
        idx = 0
        
        for file_name in files:
            
            if file_name.endswith('.json'):
                
                idx += 1
                
                input_json = os.path.join(label_dir, file_name)
                output_dir = label_dir
                
                # Gera versões aumentadas do JSON
                generate_augmented_versions(input_json, output_dir) 
                
            # Exibe o progresso (sobrescreve a mesma linha)
            print(f"Processando amostra {idx + 1} / {total} ({((idx+1)/total)*100:.2f}%)", end = "\r")