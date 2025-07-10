import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from exercise_dataset import json_to_data, LABEL_MAP, INV_LABEL_MAP

# Modelo GCN Temporal
class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 3):
        super(TemporalGCN, self).__init__()
        self.convs = nn.ModuleList()
        
        # Camada de entrada
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Camadas intermediárias
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Camada de saída
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.lstm = nn.LSTM(out_channels, out_channels, batch_first = True)
        
        # Adicionando mecanismo de atenção multi-head
        self.attention = nn.MultiheadAttention(embed_dim = out_channels, num_heads = 6, batch_first = True)

    def forward(self, x, edge_index, batch):
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        batch_size = batch.max().item() + 1
        
        seq_len = x.shape[0] // batch_size
        x = x.view(batch_size, seq_len, x.size(-1))
        lstm_out, _ = self.lstm(x)
        
        # Aplicar atenção multi-head sobre a sequência
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Agregação via média sobre a dimensão temporal
        out = attn_out.mean(dim = 1)
        
        return out

# Classe de treinamento com suporte a CUDA
class TemporalGCNClassifier:
    def __init__(self,  out_channels, in_channels = 2, hidden_channels = 64, lr = 0.001, class_weights = None, early_stop_patience = 99, lr_scheduler_patience = 99):
        
        # Define o dispositivo: CUDA se disponível, senão CPU
        if torch.cuda.is_available():
            print("CUDA disponível. Usando GPU.")
            self.device = torch.device("cuda")
        else:
            print("CUDA não disponível. Usando CPU.")
            print(torch.version.cuda) # Se isso retornar None, tem que instalar o CUDA ou validar a versão do Torch
            self.device = torch.device("cpu")
            raise Exception("CUDA não disponível. Treinamento em CPU pode ser lento.")

        self.model     = TemporalGCN(in_channels, hidden_channels, out_channels).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight = class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.early_stop_patience   = early_stop_patience
        self.lr_scheduler_patience = lr_scheduler_patience

    def train_step(self, data):
        
        # Move os dados para o dispositivo
        data = data.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        
        out = self.model(x, edge_index, batch)
        loss = self.criterion(out, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_loop(self, train_loader, val_loader, num_epochs = 30):
        train_losses     = []
        train_accuracies = []
        val_losses       = []
        val_accuracies   = []
        
        # Variáveis de controle para early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # Inicializa o scheduler ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.5,
                                                               patience = self.lr_scheduler_patience, verbose = True)

        epoch_counter   = 0  # Para retonar as epochs antes do break por early stopping
        training_epochs = "" # Prints dos dados de cada época para salvar em arquivo
        
        # Loop real de treinamento
        for epoch in range(num_epochs):
            
            epoch_counter = epoch + 1
            
            self.model.train()
            total_loss = 0
            
            for data in train_loader:
                loss = self.train_step(data)
                total_loss += loss
                
            avg_train_loss = total_loss / len(train_loader)
            self.model.eval()
            train_correct = 0
            train_total = 0
            
            with torch.no_grad():
                for data in train_loader:
                    data = data.to(self.device)
                    out  = self.model(data.x, data.edge_index, data.batch)
                    
                    predictions = out.argmax(dim = 1)
                    
                    train_correct += (predictions == data.y).sum().item()
                    train_total   += data.y.size(0)
                    
            train_accuracy = train_correct / train_total
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                
                for data in val_loader:
                    data = data.to(self.device)
                    out  = self.model(data.x, data.edge_index, data.batch)
                    loss = self.criterion(out, data.y)
                    
                    total_val_loss += loss.item()
                    predictions    = out.argmax(dim = 1)
                    val_correct    += (predictions == data.y).sum().item()
                    val_total      += data.y.size(0)
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            train_losses    .append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            val_losses      .append(avg_val_loss)
            val_accuracies  .append(val_accuracy)
            
            epoch_string = f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%"
            print(epoch_string)
            training_epochs += epoch_string + "\n"
            
            # Reduz a taxa de aprendizado se a loss de validação não melhorar
            scheduler.step(avg_val_loss)
            
            # Early stopping: verifica se a loss de validação melhorou
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping ativado: Val Loss não melhorou por {self.early_stop_patience} épocas consecutivas.")
                break   
        
        return train_losses, train_accuracies, val_losses, val_accuracies, epoch_counter, training_epochs

    def predict(self, json_path):
        
        # Carrega e pré-processa os keypoints
        data = json_to_data(json_path)
        
        # Cria um batch com uma única amostra
        batch_data = Batch.from_data_list([data])
        
        data = batch_data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim = 1).item()
            
        return INV_LABEL_MAP[pred]
