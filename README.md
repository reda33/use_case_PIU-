# use_case_PIU-

Projet : Analyse Non Supervisée de l'Usage Problématique d'Internet

 Description du Projet

Ce projet vise à analyser l'usage problématique d'Internet en utilisant des techniques de modélisation non supervisée. L’objectif est d'identifier des groupes d’utilisateurs ayant des comportements similaires et de détecter des anomalies dans les habitudes d'utilisation.

Étapes du Projet

1️⃣ Exploration et Prétraitement des Données

Chargement des données provenant du Child Mind Institute.

Nettoyage des valeurs manquantes et gestion des outliers.

Normalisation des données pour garantir une cohérence des valeurs.

2️⃣ Feature Engineering

Création de nouvelles variables (ex : catégories d’âge, indicateurs d’utilisation excessive d’Internet).

Encodage des variables catégorielles.

Transformation et standardisation des variables numériques.

3️⃣ Modélisation Non Supervisée

Réduction de dimension : PCA et t-SNE pour mieux visualiser les données.

Clustering : Utilisation de HDBSCAN pour segmenter les utilisateurs en groupes distincts.

Détection des anomalies : Méthodes comme Isolation Forest et LOF pour repérer les comportements atypiques.

4️⃣ Analyse et Interprétation

Étude des caractéristiques de chaque groupe identifié.

Validation des résultats avec des métriques comme le Silhouette Score.

Visualisation des clusters et des anomalies.

5️⃣ Conclusions et Recommandations

Présentation des insights clés issus de l’analyse.

Proposition d’actions pour la prévention de l’usage problématique d’Internet.

Développement potentiel d’un outil d’alerte basé sur ces analyses.

📁 Organisation des Fichiers

📂 data/                # Contient les datasets utilisés et transformés
    ├── train.csv        # Dataset brut
    ├── train_cleaned.csv # Dataset nettoyé
    ├── train_ready.csv   # Dataset après Feature Engineering
    ├── train_clustered.csv # Résultats après clustering
    ├── train_analyzed.csv  # Résultats après détection des anomalies

📂 notebooks/            # Contient les notebooks Jupyter
    ├── EDA.ipynb         # Analyse exploratoire des données
    ├── Feature_Engineering.ipynb # Transformation des données
    ├── Modelisation_Non_Supervisee.ipynb # Clustering et anomalies


📂 src/                 # Contient les scripts Python utilisés
    ├── data_cleaning.py  # Nettoyage des données
    ├── feature_engineering.py  # Transformation des variables
    ├── clustering.py     # Modélisation non supervisée

Technologies Utilisées

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

Machine Learning : PCA, t-SNE, HDBSCAN, Isolation Forest, LOF

Jupyter Notebook pour la visualisation et l’expérimentation

Pdf, PowerPoint pour la restitution des résultats

Prochaines Étapes

Approfondir l'analyse des clusters pour mieux comprendre leurs différences.

Tester d'autres méthodes de réduction de dimension (Autoencodeurs).

Implémenter une approche semi-supervisée pour améliorer les résultats.

Auteur : Rida Khayi  Date : 10 mars 2025


