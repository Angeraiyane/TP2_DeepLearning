# TP2 : Optimisation et Régularisation des FFN
##  Notes Spéciales sur l'Exécution (Exercice 5)

En raison d'un conflit d'importation mineur survenu à la toute fin de l'exécution de l'exploration aléatoire (`ModuleNotFoundError` sur la variable du Grid Search alors que les 48 tirages du Random Search étaient déjà complétés avec succès), le flux de travail a été sécurisé et partitionné pour éviter une ré-exécution coûteuse en temps de calcul :

1. **`Exercice5.py` (Script d'origine) :** Contient le pipeline complet d'entraînement brut. C'est ce script qui a généré les logs visibles sur la console et qui a validé le TOP 10 des configurations optimales.
2. **`sauve_rs.py` :** Un script d'urgence créé immédiatement après l'interruption pour dumper et figer l'ensemble des 48 tirages de la console vers un fichier structuré de sauvegarde locale (`random_search_results.csv`).
3. **`sauver_exercice5.py` :** Une version optimisée et autonome qui charge directement le fichier `random_search_results.csv` pour générer instantanément l'analyse descriptive avancée, les graphiques de dispersion (`rs_scatter_analytics.png`), ainsi que les corrélations de Spearman associées sans aucune altération des données d'origine.

Cette journalisation atteste de l'intégrité des données recueillies et de la continuité logique entre la phase d'optimisation et la phase de rapport de synthèse.
