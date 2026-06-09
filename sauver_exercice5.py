import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement de tes vrais résultats sauvegardés
try:
    rs_results = pd.read_csv('random_search_results.csv')
    print("Données chargées avec succès depuis le fichier CSV ! \n")
except FileNotFoundError:
    # Option de secours immédiate avec tes exactes données copiées
    data = {
        'hidden_dims': ['[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[64, 32]', '[256, 128, 64, 32]', '[256, 128, 64, 32]', '[128, 64, 32]'],
        'activation': ['leaky_relu', 'relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'relu', 'relu', 'leaky_relu', 'relu'],
        'dropout_rate': [0.012629, 0.158158, 0.341642, 0.022749, 0.080803, 0.065537, 0.117363, 0.250979, 0.272801, 0.424696],
        'lr': [0.000817, 0.001641, 0.001593, 0.002336, 0.000521, 0.000407, 0.002592, 0.000365, 0.000599, 0.001674],
        'weight_decay': [0.002267, 0.000032, 0.001331, 0.000209, 0.000022, 0.000020, 0.001538, 0.000425, 0.000065, 0.000026],
        'val_mse': [0.266311, 0.268800, 0.279573, 0.281652, 0.283054, 0.291697, 0.292283, 0.294021, 0.295397, 0.308008]
    }
    rs_results = pd.DataFrame(data)
    print("Données reconstruites depuis le dictionnaire d'urgence ! \n")

# 2. Génération instantanée du graphique de l'Analyse Descriptive (Q5)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x='lr', y='val_mse', hue='dropout_rate', data=rs_results, ax=axs[0], palette='coolwarm')
axs[0].set_xscale('log')
axs[0].set_title('val_mse en fonction de LR et Dropout')

sns.scatterplot(x='weight_decay', y='val_mse', data=rs_results, ax=axs[1], color='purple')
axs[1].set_xscale('log')
axs[1].set_title('val_mse en fonction du Weight Decay')

plt.tight_layout()
plt.savefig('rs_scatter_analytics.png', dpi=120)
print("-> Graphique 'rs_scatter_analytics.png' sauvegardé !")
plt.show()

# 3. Calcul instantané des corrélations de Spearman (Q6)
print("\n=== CORRÉLATION DE SPEARMAN AVEC VAL_MSE ===")
for col in ['lr', 'weight_decay', 'dropout_rate']:
    corr = rs_results[col].corr(rs_results['val_mse'], method='spearman')
    print(f"Corrélation de Spearman pour '{col}' : {corr:.4f}")