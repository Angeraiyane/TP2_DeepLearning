
#Exercice 1 : Exploration et Préparation du Dataset California Housing
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Fixer les seeds pour la reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# --- Partie 1.1 : Chargement et analyse exploratoire ---
# Q1: Chargement et statistiques
california = fetch_california_housing(as_frame=True)
df = california.frame

print("=== STATISTIQUES DESCRIPTIVES ===")
print(f"Nombre d'exemples : {df.shape[0]}")
print(f"Nombre de features : {len(california.feature_names)}")
print(f"Noms des features : {california.feature_names}\n")
print(df.head())
print(df.describe())

# Q2: Distribution de la target (MedHouseVal)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['MedHouseVal'], bins=50, kde=True, color='blue')
plt.title('Distribution de MedHouseVal')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['MedHouseVal'], color='cyan')
plt.title('Boxplot de MedHouseVal')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=120)
plt.show()

# Commentaire Q2 : On remarque un pic artificiel franc à la valeur 5.0. 
# La variable cible a été plafonnée lors de la collecte des données.

# Q3: Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation de Pearson')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=120)
plt.show()

# MedInc' (revenu médian) est le plus corrélé à la cible.
# 'AveRooms' et 'AveBedrms' présentent la plus forte colinéarité.


# Q4: Split 70% Train / 15% Val / 15% Test avec stratification
# Création de bins sur la target pour une stratification propre en régression
df['target_bins'] = pd.cut(df['MedHouseVal'], bins=5, labels=False)

X = df.drop(columns=['MedHouseVal', 'target_bins']).values
y = df['MedHouseVal'].values.reshape(-1, 1)
bins = df['target_bins'].values

# Premier split : 70% Train et 30% Reste (Val + Test)
X_train, X_temp, y_train, y_temp, _, bins_temp = train_test_split(
    X, y, bins, test_size=0.30, random_state=42, stratify=bins
)

# Second split : Séparation des 30% restants en deux parts égales (15% Val, 15% Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=bins_temp
)

########---PARTIE 2 : Préparation des données pour PyTorch---########

# Q5: Normalisation StandardScaler
# On applique fit_transform uniquement sur Train pour éviter le Data Leakage (fuite de données)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Q6: Création des TensorDatasets & DataLoaders
train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Vérification des dimensions
xb, yb = next(iter(train_loader))
print(f'\nX batch shape : {xb.shape} | y batch shape : {yb.shape}')
print(f'X mean: {xb.mean().item():.4f} | X std: {xb.std().item():.4f}')
