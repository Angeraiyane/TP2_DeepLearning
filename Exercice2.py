
# ### Exercice 2 - Implémentation du MLP et Entraînement de la Baseline

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Q1 & Q2: Définition de la classe modulaire DeepFFN
class DeepFFN(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dims: list = [128, 64, 32], 
                 output_dim: int = 1, activation: str = 'relu', 
                 use_bn: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        
        self.activation_name = activation.lower()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.get_activation(self.activation_name))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.init_weights()

    def get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name, nn.ReLU())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ['relu', 'leaky_relu', 'elu']:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output_layer(self.layers(x))

# Q4: Fonction train_one_epoch avec tracking du gradient avant clipping
def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss, total_gnorm, n = 0.0, 0.0, 0
    
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        # Récupérer la norme globale L2 du gradient AVANT clipping pour monitoring
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        total_gnorm += grad_norm.item()
        
        # Application effective du gradient clipping
        if clip_value is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
        
    return total_loss / n, total_gnorm / len(loader)

# Q5: Fonction d'évaluation calculant MSE, MAE et R²
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    n = 0
    
    for xb, yb in loader:
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * len(xb)
        n += len(xb)
        preds.append(out)
        targets.append(yb)
        
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    
    return mse, mae, r2

# Q6: Boucle d'entraînement complète avec Early Stopping et Plateau Scheduler
def train_model(model, train_loader, val_loader, config):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config.get('weight_decay', 0.0)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    clip_value = config.get('clip_value', 1.0)
    
    history = {
        'train_mse': [], 'val_mse': [], 'val_mae': [], 
        'val_r2': [], 'lr': [], 'grad_norm': []
    }
    
    best_val = float('inf')
    patience = config.get('early_stopping_patience', 20)
    no_improve = 0
    t0 = time.time()
    
    for epoch in range(config.get('epochs', 200)):
        tr_loss, gnorm = train_one_epoch(model, train_loader, optimizer, criterion, clip_value)
        va_mse, va_mae, va_r2 = evaluate(model, val_loader, criterion)
        
        scheduler.step(va_mse)
        
        history['train_mse'].append(tr_loss)
        history['val_mse'].append(va_mse)
        history['val_mae'].append(va_mae)
        history['val_r2'].append(va_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norm'].append(gnorm)
        
        if va_mse < best_val:
            best_val = va_mse
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f' Early stopping à l\'epoch {epoch+1}')
            break
            
        if (epoch + 1) % 20 == 0:
            print(f'Ep [{epoch+1:3d}] | tr MSE={tr_loss:.4f} | val MSE={va_mse:.4f} | R²={va_r2:.4f} | lr={optimizer.param_groups[0]["lr"]:.6f}')
            
    return history, best_val, time.time() - t0

# %%
# Q3 & Q7: Instanciation et exécution de la Baseline
if __name__ == '__main__':
    from Exercice1 import train_loader, val_loader
    
    model = DeepFFN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres entraînables : {total_params}')
    
    config_baseline = {
        'hidden_dims': [128, 64, 32],
        'activation': 'relu',
        'use_bn': True,
        'dropout_rate': 0.2,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'clip_value': 1.0,
        'epochs': 200,
        'early_stopping_patience': 25
    }
    
    print("\n--- Entraînement du modèle Baseline ---")
    history, best_val, elapsed = train_model(model, train_loader, val_loader, config_baseline)
    
    # Plot des courbes de performance de la baseline (5 sous-graphes)
    epochs_range = range(1, len(history['train_mse']) + 1)
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    axs[0].plot(epochs_range, history['train_mse'], label='Train MSE')
    axs[0].plot(epochs_range, history['val_mse'], label='Val MSE')
    axs[0].set_title('MSE Loss')
    axs[0].legend()
    
    axs[1].plot(epochs_range, history['val_mae'], color='orange')
    axs[1].set_title('Validation MAE')
    
    axs[2].plot(epochs_range, history['val_r2'], color='green')
    axs[2].set_title('Validation R²')
    
    axs[3].plot(epochs_range, history['lr'], color='red')
    axs[3].set_title('Learning Rate Decay')
    
    axs[4].plot(epochs_range, history['grad_norm'], color='purple')
    axs[4].set_title('Gradient Norm (Avant Clipping)')
    
    plt.tight_layout()
    plt.savefig('baseline_curves.png', dpi=120)
    plt.show()