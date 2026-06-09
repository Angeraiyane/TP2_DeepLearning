# %% [markdown]
# ### Exercice 5 - Recherche Aléatoire Continue (Random Search)

# %%
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from Exercice1 import train_loader, val_loader
from Exercice2 import DeepFFN, train_model

# Q1: Définition de l'espace de recherche continu et échantillonnage
search_space = {
    'lr':           ('log_uniform', 1e-4, 1e-2),
    'weight_decay': ('log_uniform', 1e-5, 1e-2),
    'dropout_rate': ('uniform', 0.0, 0.5),
    'clip_value':   ('uniform', 0.5, 5.0),
    'hidden_dims':  ('choice', [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64], [256, 128, 64, 32]]),
    'activation':   ('choice', ['relu', 'leaky_relu', 'elu', 'selu'])
}

def sample_config(space):
    config = {}
    for key, spec in space.items():
        dist = spec[0]
        if dist == 'log_uniform':
            config[key] = float(np.exp(np.random.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif dist == 'uniform':
            config[key] = float(random.uniform(spec[1], spec[2]))
        elif dist == 'choice':
            config[key] = random.choice(spec[1])
    return config

# Q2: Implémentation du Random Search
def random_search(space, n_trials, train_loader, val_loader, epochs=80):
    results = []
    print(f'Random Search : {n_trials} tirages aléatoires ({epochs} epochs max).')
    print('-' * 60)
    
    for trial in range(n_trials):
        config = sample_config(space)
        config['epochs'] = epochs
        config['early_stopping_patience'] = 15
        config['use_bn'] = True
        
        torch.manual_seed(trial)  # Une seed changeant par essai unique !
        
        model = DeepFFN(
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            dropout_rate=config['dropout_rate']
        )
        
        best_mse, elapsed = train_model(model, train_loader, val_loader, config)[1:]
        
        config_record = config.copy()
        config_record['hidden_dims'] = str(config_record['hidden_dims'])
        
        results.append({**config_record, 'val_mse': best_mse, 'trial': trial, 'time_s': elapsed})
        print(f'Trial [{trial+1:3d}/{n_trials}] val MSE={best_mse:.4f} | lr={config["lr"]:.2e} | act={config["activation"]}')
        
    df = pd.DataFrame(results).sort_values('val_mse')
    return df

# %%
if __name__ == '__main__':
    # Q3: Lancement du Random Search
    rs_results = random_search(search_space, n_trials=48, train_loader=train_loader, val_loader=val_loader, epochs=80)
    
    print('\n=== TOP 10 CONFIGURATIONS RANDOM SEARCH ===')
    print(rs_results[['hidden_dims', 'activation', 'dropout_rate', 'lr', 'weight_decay', 'val_mse']].head(10).to_string(index=False))
    
    # --- DEBUT DE LA REPRISE SUR ERREUR ---
    # Q4: Courbe de convergence (Best-so-far plot) avec le bon nom d'import
    try:
        from Exercice4 import gs_results
        
        # On trie chronologiquement selon l'ordre originel de l'évaluation
        gs_ordered = gs_results.sort_index() 
        rs_ordered = rs_results.sort_values('trial') if 'trial' in rs_results.columns else rs_results
        
        gs_best_sofar = np.minimum.accumulate(gs_ordered['val_mse'].values)
        rs_best_sofar = np.minimum.accumulate(rs_ordered['val_mse'].values)
        
        plt.figure(figsize=(9, 5))
        plt.plot(gs_best_sofar, label='Grid Search (Discret)', marker='o')
        plt.plot(rs_best_sofar, label='Random Search (Continu)', marker='s')
        plt.xlabel('Nombre de configurations évaluées')
        plt.ylabel('Meilleur val MSE trouvé')
        plt.title('Convergence Comparative : Grid Search vs Random Search')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('gs_vs_rs.png', dpi=120)
        plt.show()
    except Exception as e:
        print(f"Note : Impossible de tracer le graphe comparatif immédiatement ({e}).")
        print("Assure-toi que Exercice4.py a bien fini de tourner et contient la variable 'gs_results'.")

    # Q5 & Q6: Analyse descriptive avancée et corrélation de Spearman
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(x='lr', y='val_mse', hue='dropout_rate', data=rs_results, ax=axs[0], palette='coolwarm')
    axs[0].set_xscale('log')
    axs[0].set_title('val_mse en fonction de LR et Dropout')
    
    sns.scatterplot(x='weight_decay', y='val_mse', data=rs_results, ax=axs[1], color='purple')
    axs[1].set_xscale('log')
    axs[1].set_title('val_mse en fonction du Weight Decay')
    plt.tight_layout()
    plt.savefig('rs_scatter_analytics.png', dpi=120)
    plt.show()
    
    print("\n=== CORRÉLATION DE SPEARMAN AVEC VAL_MSE ===")
    for col in ['lr', 'weight_decay', 'dropout_rate']:
        if col in rs_results.columns:
            corr = rs_results[col].corr(rs_results['val_mse'], method='spearman')
            print(f"Corrélation de Spearman pour '{col}' : {corr:.4f}")