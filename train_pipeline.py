import torch
import os
import shutil
import random
import sys
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from exercise_dataset import ExerciseDataset, LABEL_MAP, INV_LABEL_MAP
from temporal_gcn import TemporalGCNClassifier
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Configurações e Paths
DATASET_PATH = "output" # Pasta com os JSONs organizados por classe
TRAIN_PATH   = "dataset/train"
VAL_PATH     = "dataset/val"

BATCH_SIZE      = 8
EPOCHS          = 200 # Número máximo de épocas para o treinamento - Normalmente, o early stopping interrompe antes
CHECKPOINT_PATH = "train_results" # Salva os resultados do treinamento (modelos, gráficos, etc.)

# Parâmetros para early stopping e scheduler
EARLY_STOP_PATIENCE   = 6  # Número de épocas sem melhora para interromper o treinamento
LR_SCHEDULER_PATIENCE = 4  # Número de épocas sem melhora para reduzir a taxa de aprendizado

# Divide dinamicamente os vídeos do dataset entre treino e validação.
def split_dataset(dataset_path, train_path, val_path, train_ratio = 0.8):
    
    if os.path.exists(train_path) or os.path.exists(val_path):
        print("Dataset já dividido.")
        return
    
    os.makedirs(train_path)
    os.makedirs(val_path)

    for label_name in os.listdir(dataset_path):
        
        class_dir = os.path.join(dataset_path, label_name)
        if not os.path.isdir(class_dir):
            continue

        train_label_dir = os.path.join(train_path, label_name)
        val_label_dir   = os.path.join(val_path, label_name)

        os.makedirs(train_label_dir, exist_ok = True)
        os.makedirs(val_label_dir  , exist_ok = True)

        files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        random.shuffle(files)

        train_size = int(len(files) * train_ratio)

        train_files = files[:train_size]
        val_files = files[train_size:]

        for file in train_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(train_label_dir, file))

        for file in val_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(val_label_dir, file))

    print(f"Dataset dividido: {train_ratio * 100}% treino, {100 - train_ratio * 100}% validação.")
    
# Gera dados de avaliação detalhados do modelo, incluindo matriz de confusão e erros de classificação.
def detailed_model_eval(classifier, dataset, save_folder = None, timestamp = None, operation = None):
    
    # Crie um DataLoader para avaliação, com batch_size = 1 para facilitar a extração dos nomes
    eval_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    
    pred_labels   = []
    real_labels   = []
    all_filenames = []
    
    classifier.model.eval()
    
    idx_item = 0
    
    save_data = save_folder is not None and timestamp is not None
    
    with torch.no_grad():
        for item in eval_loader:
            
            data = item 
            filepath = dataset.get_item_path(idx_item)
            
            data = data.to(classifier.device)
            out = classifier.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim = 1).item()
            real = data.y.item()
            
            pred_labels.append(pred)
            real_labels.append(real)

            all_filenames.append(filepath)
            
            if operation == "3" and pred != real:
                print(f"Erro: {filepath:<100} Pred: {INV_LABEL_MAP[pred]:<25} | Real {INV_LABEL_MAP[real]:<25}")

            idx_item += 1
    
    all_possible_indices = list(range(len(LABEL_MAP)))
    ordered_labels = [label for label, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
    
    # Calcular a Matriz de confusão
    cm = confusion_matrix(real_labels, pred_labels, labels = all_possible_indices)
    
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ordered_labels)

    fig = plt.figure(figsize = (12, 12))
    disp.plot(cmap = plt.cm.Blues)
    plt.title("Matriz de confusão")
    plt.ylabel("Classe real")
    plt.xlabel("Classe estimada")
    plt.xticks(rotation = 90)
    plt.tight_layout()

    if save_data:
        cm_filename = os.path.join(save_folder, f"model_{timestamp}_confusion_matrix.png")
        plt.savefig(cm_filename, bbox_inches = "tight", pad_inches = 0.1)
        print(f"Matriz de confusão salva em {cm_filename}")
        
        # Gerar um relatório de erros: para cada erro, salvar o nome do arquivo, classe esperada e classe estimada
    
        # Mapeamento inverso para as classes (para escrever no arquivo)
        inv_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        
        error_filename = os.path.join(save_folder, f"model_{timestamp}_classification_errors.txt")
        with open(error_filename, "w") as f:
            f.write("Erros:\n")
            
            for i in range(len(real_labels)):
                if pred_labels[i] != real_labels[i]:
                    true_class = inv_LABEL_MAP.get(real_labels[i])
                    pred_class = inv_LABEL_MAP.get(pred_labels[i])
                    
                    f.write(f"Arquivo: {all_filenames[i]}, Real: {true_class}, Estimada: {pred_class}\n")
                    
        print(f"Erros salvos em {error_filename}")
    else:
        plt.show()
        plt.close(fig)

# Pipeline de treinamento do modelo
# Carrega o dataset, treina o modelo e salva os resultados
def train_model():
    
    # Divisão dinâmica do Dataset
    split_dataset(DATASET_PATH, TRAIN_PATH, VAL_PATH, train_ratio = 0.8)

    #  Carregar os datasets de treino e validação
    train_dataset = ExerciseDataset(TRAIN_PATH, cache = True)
    val_dataset   = ExerciseDataset(VAL_PATH  , cache = True)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True )
    val_loader   = DataLoader(val_dataset  , batch_size = BATCH_SIZE, shuffle = False)
    
    # Cálculo dos pesos de classes com base na frequência das amostras
    labels = [LABEL_MAP[label] for (_, label) in train_dataset.samples]
    unique, counts = np.unique(labels, return_counts = True)
    class_weights  = torch.tensor(1.0 / counts, dtype = torch.float32)
    class_weights  = class_weights / class_weights.sum() * len(unique)

    # Criar o Modelo
    classifier = TemporalGCNClassifier(in_channels = 50, hidden_channels = 128, out_channels = len(LABEL_MAP), class_weights = class_weights,
                                       early_stop_patience = EARLY_STOP_PATIENCE, lr_scheduler_patience = LR_SCHEDULER_PATIENCE)

    # Treinamento com Métricas via train_loop da Classe
    train_losses, train_accuracies, val_losses, val_accuracies, epoch_counter, training_epochs = classifier.train_loop(train_loader, val_loader, num_epochs = EPOCHS)


    # Gerar um timestamp e criar uma pasta para salvar os resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(CHECKPOINT_PATH, f"model_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)
    
    epochs_filename = os.path.join(save_folder, f"model_{timestamp}_training_epochs.txt")
    with open(epochs_filename, "w") as f:
        f.write(training_epochs)

    # Salvar os pesos do modelo
    model_filename = os.path.join(save_folder, f"model_{timestamp}.pth")
    torch.save(classifier.model.state_dict(), model_filename)
    print(f"Modelo salvo em {model_filename}")
    
    print(f"Modelo salvo em { CHECKPOINT_PATH }")
    
    # Cores para cada métrica
    color_train_loss = 'tab:red'
    color_val_loss   = 'tab:orange'
    color_train_acc  = 'tab:blue'
    color_val_acc    = 'tab:green'

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo 1 para Loss
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Loss", color=color_train_loss)
    l1, = ax1.plot(range(1, epoch_counter + 1), train_losses, label = "Train Loss", color = color_train_loss)
    l2, = ax1.plot(range(1, epoch_counter + 1), val_losses,   label = "Val Loss"  , color = color_val_loss  )
    ax1.tick_params(axis = 'y', labelcolor=color_train_loss)

    # Criar um segundo eixo y para Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color=color_train_acc)
    
    l3, = ax2.plot(range(1, epoch_counter + 1), [acc * 100 for acc in train_accuracies], 
                label="Train Accuracy", color = color_train_acc)
    
    l4, = ax2.plot(range(1, epoch_counter + 1), [acc * 100 for acc in val_accuracies], 
                label = "Val Accuracy", color = color_val_acc)
    
    ax2.tick_params(axis = 'y', labelcolor=color_train_acc)

    # Combina todas as linhas para criar uma única legenda
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    plt.title("Loss e Accuracy vs. Épocas")
    plt.legend(lines, labels, loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol=2)

    plt.tight_layout()
    
    graph_filename = os.path.join(save_folder, f"model_{timestamp}_graph.png")
    plt.savefig(graph_filename)
    print(f"Grafico de treinamento salvo em {graph_filename}")

    detailed_model_eval(classifier, val_dataset, save_folder, timestamp, operation = operation)

if __name__ == "__main__":
    
    # Processar argumentos
    n = len(sys.argv)
    if n < 2:
        print ("Informe o tipo de operação 1 - Treinar | 2 - Avaliar existente | 3 - Avaliar com dados externos\n")
        print ("Ou informe 'all' para todos.\n")
        sys.exit(1)
        
    operation = sys.argv[1]
    print ("Operação: ", operation)
    
    # Treinamento de um novo modelo do zero
    if (operation == "1"):
        print("Treinando modelo...")
        train_model()
        sys.exit()
    
    # Avaliação de um modelo existente
    if operation == "2" or operation == "3":
        print("Avaliar modelo existente...")
        
        if n < 3:
            print ("Informe o nome do modelo\n")
            sys.exit(1)
            
        model_name = sys.argv[2]
        
        path_folder = os.path.join(CHECKPOINT_PATH, model_name)
        if not os.path.exists(path_folder):
            print(f"Modelo {model_name} não encontrado em {CHECKPOINT_PATH}.")
            sys.exit(1)
        
        # Carrega o Modelo
        classifier = TemporalGCNClassifier(in_channels = 50, hidden_channels = 128, out_channels = len(LABEL_MAP))
        
        model = torch.load(os.path.join(path_folder, f"{model_name}.pth"))
        classifier.model.load_state_dict(model)
        
        exercise_dataset_path = DATASET_PATH if operation == "2" else "output_externo"
        dataset = ExerciseDataset(exercise_dataset_path, cache = True, aug = False)
        detailed_model_eval(classifier, dataset, operation = operation)
        
        sys.exit()
