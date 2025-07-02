# Diagnostic CRM vs BI

Cet outil Streamlit analyse la correspondance entre les données issues d'un CRM et celles provenant de la BI. Le script `streamlit_app.py` permet de comparer un ou plusieurs fichiers de données et de visualiser plusieurs indicateurs de qualité à l'aide de graphiques interactifs.

## Fonctionnement général

1. **Choix du mode d'analyse** :
   - **Analyse Simple** pour traiter un seul fichier (max 200 MB).
   - **Analyse Fragmentée** pour combiner automatiquement plusieurs fichiers CSV ou Excel, même si leurs structures diffèrent.
2. **Chargement des données** : les fichiers sont lus puis combinés en une table unique.
3. **Calcul des indicateurs** : pour chaque champ *_IsEquals*, le taux de match, le nombre d'écarts et de valeurs nulles sont calculés.
4. **Affichage des résultats** : différentes sections de l'application présentent les résultats sous forme de métriques et de graphiques.

## Graphiques et tableaux affichés

- **Métriques globales** :
  - Taux de correspondance global et totaux (correspondances, valeurs nulles, écarts, etc.).
  - Camembert détaillé de la répartition des correspondances et écarts.
- **Analyse détaillée des champs** :
  - Histogramme du taux de correspondance par champ.
  - Histogramme empilé indiquant le nombre de correspondances, d'écarts et de valeurs nulles pour chaque champ.
- **Analyses avancées** :
  - Répartition par champ et type de situation (valeurs différentes, nulles côté CRM ou BI…).
  - Matrice de priorisation (impact métier vs complexité de correction) pour identifier les champs à traiter en priorité.
- **Analyse champ par champ** *(optionnelle)* :
  - Tableau d'exemples de lignes en écart pour chaque champ sélectionné avec des métriques précises.

Chaque tableau ou graphique est généré grâce à Plotly et mis en cache 


## Tuto

1. Choisissez le mode **Analyse Simple** ou **Analyse Fragmentée** dans la barre latérale.
2. Téléversez vos fichiers CSV ou Excel.
3. Sélectionnez les options d'analyse et cliquez sur **"Appliquer l'analyse"**.
4. Consultez les métriques et explorez les graphiques interactifs.

Lors du premier lancement, le calcul peut prendre un peu de temps. Les données et les graphiques sont ensuite mis en cache pour les exécutions suivantes.

## Personnalisation

Le thème de WYZ est défini dans [`.streamlit/config.toml`](.streamlit/config.toml).
