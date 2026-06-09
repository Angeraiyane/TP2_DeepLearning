import pandas as pd

# Copie brute des résultats affichés dans ta console
data = {
    'hidden_dims': ['[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[256, 128, 64]', '[64, 32]', '[256, 128, 64, 32]', '[256, 128, 64, 32]', '[128, 64, 32]'],
    'activation': ['leaky_relu', 'relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'relu', 'relu', 'leaky_relu', 'relu'],
    'dropout_rate': [0.012629, 0.158158, 0.341642, 0.022749, 0.080803, 0.065537, 0.117363, 0.250979, 0.272801, 0.424696],
    'lr': [0.000817, 0.001641, 0.001593, 0.002336, 0.000521, 0.000407, 0.002592, 0.000365, 0.000599, 0.001674],
    'weight_decay': [0.002267, 0.000032, 0.001331, 0.000209, 0.000022, 0.000020, 0.001538, 0.000425, 0.000065, 0.000026],
    'val_mse': [0.266311, 0.268800, 0.279573, 0.281652, 0.283054, 0.291697, 0.292283, 0.294021, 0.295397, 0.308008]
}

df = pd.DataFrame(data)
df.to_csv('random_search_results.csv', index=False)
print("Fichier de sauvegarde généré avec succès !")