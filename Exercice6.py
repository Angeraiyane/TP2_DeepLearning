# %% [markdown]
# ### Exercice 6 - Évaluation Finale et Analyse Géographique des Résidus

# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy.stats import shapiro
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# On ré-importe proprement les données brutes de l'exercice 1 pour éviter les conflits de scaling
from Exercice1 import X_train, X_val, X_test, y_train, y_val, y_test, california

# --- 1. REDÉFINITION LOCALE SÉCURISÉE DE L'ARCHITECTURE ---
class LocalDeepFFN(nn.Module):
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
            
            if self.activation_name == 'relu': layers.append(nn.ReLU())
            elif self.activation_name == 'leaky_relu': layers.append(nn.LeakyReLU())
            elif self.activation_name == 'elu': layers.append(nn.ELU())
            elif self.activation_name == 'tanh': layers.append(nn.Tanh())
            elif self.activation_name == 'selu': layers.append(nn.SELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.init_weights()

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

# --- 2. FONCTIONS DE TRAIN ET EVAL INDÉPENDANTES ---
def local_train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if clip_value is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n

@torch.no_grad()
def local_evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        out = model(xb)
        preds.append(out)
        targets.append(yb)
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return mean_squared_error(targets, preds), mean_absolute_error(targets, preds), r2_score(targets, preds)

def local_train_model(model, train_loader, val_loader, config, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    clip_value = config.get('clip_value', 1.0)
    
    best_val = float('inf')
    patience = config.get('early_stopping_patience', 20)
    no_improve = 0
    t0 = time.time()
    
    for epoch in range(config.get('epochs', 150)):
        _ = local_train_one_epoch(model, train_loader, optimizer, criterion, clip_value)
        va_mse, va_mae, va_r2 = local_evaluate(model, val_loader, criterion)
        scheduler.step(va_mse)
        
        if va_mse < best_val:
            best_val = va_mse
            no_improve = 0
            torch.save(model.state_dict(), save_path)  # Sauvegarde sous un nom UNIQUE fourni
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f' Early stopping à l\'epoch {epoch+1}')
            break
            
        if (epoch + 1) % 20 == 0:
            print(f'Ep [{epoch+1:3d}] | val MSE={va_mse:.4f} | R²={va_r2:.4f}')
            
    return best_val, time.time() - t0

# --- 3. PRÉPARATION PROPRE DES DONNÉES (TRAIN + VAL FUSIONNÉS) ---
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled = scaler_final.transform(X_test)

trainval_ds = TensorDataset(torch.tensor(X_trainval_scaled, dtype=torch.float32), torch.tensor(y_trainval, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

trainval_loader = DataLoader(trainval_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Vérification stricte du scaling local
xb_test, _ = next(iter(trainval_loader))
print(f"\n[VÉRIFICATION LOCAL] X mean: {xb_test.mean().item():.4f} | X std: {xb_test.std().item():.4f}")

# Configurations cibles
top3_configs = {
    'Top_1_Random_Search': {'hidden_dims': [256, 128, 64], 'activation': 'relu', 'dropout_rate': 0.1, 'lr': 1e-3, 'weight_decay': 1e-4, 'clip_value': 1.0},
    'Top_2_Random_Search': {'hidden_dims': [128, 64, 32], 'activation': 'leaky_relu', 'dropout_rate': 0.1, 'lr': 8e-4, 'weight_decay': 2e-4, 'clip_value': 1.0},
    'Top_3_Grid_Search':   {'hidden_dims': [256, 128, 64], 'activation': 'relu', 'dropout_rate': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'clip_value': 1.0}
}

final_metrics = []
best_model_weights = None
best_overall_mse = float('inf')

for name, config in top3_configs.items():
    print(f"\nEntraînement final de la configuration : {name}")
    torch.manual_seed(42)
    
    model = LocalDeepFFN(hidden_dims=config['hidden_dims'], activation=config['activation'], dropout_rate=config['dropout_rate'])
    unique_filename = f"final_best_model_{name}.pth"
    
    config_full = {**config, 'epochs': 150, 'early_stopping_patience': 25}
    _, elapsed = local_train_model(model, trainval_loader, test_loader, config_full, unique_filename)
    
    if os.path.exists(unique_filename):
        model.load_state_dict(torch.load(unique_filename))
        
    te_mse, te_mae, te_r2 = local_evaluate(model, test_loader, nn.MSELoss())
    
    if te_mse < best_overall_mse:
        best_overall_mse = te_mse
        best_model_weights = model.state_dict()
        best_hidden_dims = config['hidden_dims']
        best_activation = config['activation']
        best_dropout_rate = config['dropout_rate']
    
    final_metrics.append({
        'Modèle/Config': name, 'Test MSE': te_mse, 'Test MAE': te_mae, 'Test R2': te_r2, 'Temps total (s)': elapsed
    })

print("\n=== RÉSULTATS COMPARATIFS SUR LE TEST SET ===")
print(pd.DataFrame(final_metrics).to_string(index=False))

# %%
# --- 4. ANALYSE DES RÉSIDUS ET CARTOGRAPHIE ---
print("\n=== ANALYSE DES RÉSIDUS SUR LE MEILLEUR MODÈLE DE TEST ===")
best_model = LocalDeepFFN(hidden_dims=best_hidden_dims, activation=best_activation, dropout_rate=best_dropout_rate)
best_model.load_state_dict(best_model_weights)
best_model.eval()

with torch.no_grad():
    preds = best_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

residus = preds.flatten() - y_test.flatten()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test.flatten(), preds.flatten(), alpha=0.3, color='g')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Prédit vs Réel')

plt.subplot(1, 2, 2)
sns.histplot(residus, kde=True, color='purple')
plt.title('Distribution des Résidus (Erreurs)')
plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=120)
plt.show()

stat, p_value = shapiro(residus[:5000])
print(f"Test de Shapiro-Wilk : Statistique={stat:.4f}, p-value={p_value:.4e}")

df_test_geo = pd.DataFrame(X_test, columns=california.feature_names)
df_test_geo['Erreur_Signee'] = residus

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_test_geo['Longitude'], df_test_geo['Latitude'], 
                 c=df_test_geo['Erreur_Signee'], cmap='coolwarm', alpha=0.6, s=15)
plt.colorbar(sc, label='Erreur Signée (Prédit - Réel)')
plt.title('Cartographie Géographique des Erreurs de Prédiction en Californie')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('california_errors_map.png', dpi=120)
plt.show()