# %% [markdown]
# ### Exercice 4 - Recherche Exhaustive par Grille (Grid Search)

# %%
import torch
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from Exercice1 import train_loader, val_loader
from Exercice2 import DeepFFN, train_model

def grid_search(param_grid, train_loader, val_loader, epochs=80):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    results = []
    
    print(f'Grid Search : {len(combos)} configurations uniques à évaluer ({epochs} epochs max).')
    print('-' * 60)
    
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        config['epochs'] = epochs
        config['early_stopping_patience'] = 15
        config['use_bn'] = True
        
        torch.manual_seed(42)  # Assurer la reproductibilité intrinsèque
        
        # Instanciation dynamique basée sur les variables de la fonction d'init
        model = DeepFFN(
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            dropout_rate=config['dropout_rate']
        )
        
        best_mse, elapsed = train_model(model, train_loader, val_loader, config)[1:]
        
        # Traitement pour conversion en string lisible pour le DataFrame final
        config_record = config.copy()
        config_record['hidden_dims'] = str(config_record['hidden_dims'])
        
        results.append({**config_record, 'val_mse': best_mse, 'time_s': elapsed})
        print(f'[{i+1:3d}/{len(combos)}] val MSE={best_mse:.4f} ({elapsed:.1f}s)')
        
    df = pd.DataFrame(results).sort_values('val_mse')
    return df

# Q3: Définition du sous-espace restreint (48 combinaisons)
param_grid_small = {
    'hidden_dims': [[64, 32], [128, 64, 32], [256, 128, 64]],
    'activation': ['relu', 'leaky_relu'],
    'dropout_rate': [0.1, 0.3],
    'lr': [1e-3, 5e-4],
    'weight_decay': [1e-4, 1e-3],
    'clip_value': [1.0]
}

if __name__ == '__main__':
    gs_results = grid_search(param_grid_small, train_loader, val_loader, epochs=80)
    
    print('\n=== TOP 10 CONFIGURATIONS GRID SEARCH ===')
    print(gs_results[['hidden_dims', 'activation', 'dropout_rate', 'lr', 'weight_decay', 'val_mse']].head(10).to_string(index=False))
    
    # Q4: Visualisations
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.boxplot(x='activation', y='val_mse', data=gs_results, ax=axs[0, 0])
    axs[0, 0].set_title('Impact de l\'Activation sur val_mse')
    
    sns.boxplot(x='dropout_rate', y='val_mse', data=gs_results, ax=axs[0, 1])
    axs[0, 1].set_title('Impact du Dropout sur val_mse')
    
    pivot = gs_results.pivot_table(index='lr', columns='weight_decay', values='val_mse', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap='viridis', ax=axs[1, 0])
    axs[1, 0].set_title('Interaction LR x Weight Decay (MSE moyen)')
    
    # Top 15 Barplot
    sns.barplot(x='val_mse', y='hidden_dims', data=gs_results.head(15), ax=axs[1, 1], palette='plasma')
    axs[1, 1].set_title('Top 15 des architectures selon val_mse')
    
    plt.tight_layout()
    plt.savefig('grid_search_analytics.png', dpi=120)
    plt.show()

    # Q5: Calcul d'influence de l'hyperparamètre par variance (standard deviation)
    print("\n=== ÉVALUATION DE L'IMPACT INDIVIDUEL VIA L'ÉCART-TYPE ===")
    for param in ['hidden_dims', 'activation', 'dropout_rate', 'lr', 'weight_decay']:
        std_dev = gs_results.groupby(param)['val_mse'].mean().std()
        print(f"Écart-type inter-groupe pour '{param}': {std_dev:.6f}")