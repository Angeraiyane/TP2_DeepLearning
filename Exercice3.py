# %% [markdown]
# ### Exercice 3 - Étude d'Ablation et Gradient Clipping

# %%
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Exercice1 import train_loader, val_loader, X_train, X_val, y_train, y_val
from Exercice2 import DeepFFN, train_model
from torch.utils.data import TensorDataset, DataLoader

# Q1 & Q2: Ablation Study
variants = {
    'A (Baseline Complet)': {'use_bn': True, 'dropout_rate': 0.2, 'weight_decay': 1e-4},
    'B (Sans BatchNorm)':   {'use_bn': False, 'dropout_rate': 0.2, 'weight_decay': 1e-4},
    'C (Sans Dropout)':     {'use_bn': True, 'dropout_rate': 0.0, 'weight_decay': 1e-4},
    'D (Sans L2)':          {'use_bn': True, 'dropout_rate': 0.2, 'weight_decay': 0.0},
    'E (Aucune Régul)':     {'use_bn': False, 'dropout_rate': 0.0, 'weight_decay': 0.0}
}

histories = {}
results_ablation = []

for name, v_cfg in variants.items():
    print(f"\nExécution de la variante : {name}")
    torch.manual_seed(42)  # Même init pour comparaison équitable
    model = DeepFFN(use_bn=v_cfg['use_bn'], dropout_rate=v_cfg['dropout_rate'])
    
    run_config = {
        'lr': 1e-3,
        'weight_decay': v_cfg['weight_decay'],
        'clip_value': 1.0,
        'epochs': 100,
        'early_stopping_patience': 15
    }
    
    hist, best_mse, elapsed = train_model(model, train_loader, val_loader, run_config)
    histories[name] = hist
    results_ablation.append({'Configuration': name, 'Best Val MSE': best_mse, 'Time (s)': elapsed})

# Comparaisons et Tracés graphiques
plt.figure(figsize=(10, 6))
for name in ['A (Baseline Complet)', 'B (Sans BatchNorm)', 'C (Sans Dropout)', 'E (Aucune Régul)']:
    plt.plot(histories[name]['val_mse'], label=name)
plt.xlabel('Epochs')
plt.ylabel('Validation MSE')
plt.title('Impact de l\'ablation des composants de régularisation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ablation_study.png', dpi=120)
plt.show()

# Print du mini-tableau récapitulatif
print("\n=== BILAN DE L'ÉTUDE D'ABLATION ===")
print(pd.DataFrame(results_ablation).to_string(index=False))

# Q3 & Q4: Effet du Gradient Clipping
clip_values = [None, 0.1, 0.5, 1.0, 5.0, 10.0]
clip_histories = {}

for cv in clip_values:
    torch.manual_seed(42)
    model = DeepFFN()
    cfg = {'lr': 1e-3, 'weight_decay': 1e-4, 'clip_value': cv, 'epochs': 50, 'early_stopping_patience': 50}
    hist, _, _ = train_model(model, train_loader, val_loader, cfg)
    clip_histories[str(cv)] = hist['grad_norm']

# Boxplot des distributions de normes de gradient
plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(clip_histories))
plt.title('Distribution des normes de gradient (avant clipping) selon la contrainte de clip_value')
plt.xlabel('clip_value')
plt.ylabel('Global Gradient Norm')
plt.savefig('gradient_clipping_distribution.png', dpi=120)
plt.show()

# Q5: Cas extrême - Entraînement sur données brutes (Sans StandardScaler)
raw_train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
raw_val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
raw_train_loader = DataLoader(raw_train_ds, batch_size=64, shuffle=True)
raw_val_loader = DataLoader(raw_val_ds, batch_size=256, shuffle=False)

print("\n--- Entraînement sur données brutes SANS Clipping ---")
torch.manual_seed(42)
model_no_clip = DeepFFN()
h_no_clip, _, _ = train_model(model_no_clip, raw_train_loader, raw_val_loader, {'lr': 1e-3, 'clip_value': None, 'epochs': 30, 'early_stopping_patience': 30})

print("\n--- Entraînement sur données brutes AVEC Clipping (clip_value=1.0) ---")
torch.manual_seed(42)
model_with_clip = DeepFFN()
h_with_clip, _, _ = train_model(model_with_clip, raw_train_loader, raw_val_loader, {'lr': 1e-3, 'clip_value': 1.0, 'epochs': 30, 'early_stopping_patience': 30})