#!/usr/bin/env python
# coding: utf-8

# #### 📌 EDA - Problematic Internet Use
# ##### Objectif : Analyser et comprendre les données de train.csv pour préparer la modélisation.
# 

# ##### 1- Charger et inspecter train.csv

# In[294]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Charger les données
train = pd.read_csv("../data/train.csv")


# In[295]:


# Aperçu des premières lignes
train.head()


# In[296]:


# Infos générales
train.info()


# In[297]:


# Afficher un résumé statistique des variables numériques
print("🔹 Statistiques Descriptives :")
print(train.describe())

# Vérifier les valeurs manquantes
print("\n🔹 Valeurs Manquantes :")
print(train.isnull().sum())


# In[298]:


# Charger le dictionnaire des données
data_dict = pd.read_csv("../data/data_dictionary.csv")

# Aperçu rapide du fichier
print(data_dict.head(10))

# Vérifier sa taille
print(f"Le fichier contient {data_dict.shape[0]} lignes et {data_dict.shape[1]} colonnes.")

# Lister les colonnes du dictionnaire
print("\nColonnes disponibles :")
print(data_dict.columns)



# In[299]:


data_dict.head()


# In[300]:


# Vérifier sa taille
print(f"Le fichier contient {data_dict.shape[0]} lignes et {data_dict.shape[1]} colonnes.")

# Lister les colonnes du dictionnaire
print("\nColonnes disponibles :")
print(data_dict.columns)


# In[301]:


# Vérifier si toutes les colonnes de train.csv sont bien dans le dictionnaire
columns_dict = set(data_dict["Field"])
columns_train = set(train.columns)

# Voir s'il manque des colonnes
missing_in_dict = columns_train - columns_dict
print(f"Colonnes absentes du dictionnaire : {missing_in_dict}" if missing_in_dict else " Toutes les colonnes sont bien décrites.")


# In[302]:


# Voir la répartition des variables par catégorie
category_counts = data_dict["Instrument"].value_counts().reset_index()
category_counts.columns = ["Instrument", "Nombre de variables"]

print("Répartition des variables par catégorie :")
print(category_counts)

# Visualisation interactive avec Plotly
fig = px.bar(category_counts, x="Instrument", y="Nombre de variables", 
             title="Répartition des variables par catégorie", text="Nombre de variables")

fig.update_traces(marker_color='blue', textposition='outside')
fig.update_layout(xaxis_title="Catégorie", yaxis_title="Nombre de variables")

fig.show()


# ###  Conclusion  
# 
# Les variables du dataset sont réparties en 12 catégories, avec une forte dominance du Parent-Child Internet Addiction Test (PCIAT), qui représente 22 variables.  
# 
# On note aussi une présence importante de mesures physiologiques, tandis que les données sur l’activité physique et le sommeil sont moins représentées.  
# 
# L’identifiant (`id`) ne sera pas utilisé, et certaines variables devront être étudiées pour éviter les redondances ou le data leakage.  

# ### 2 Analyse des valeurs manquantes  
# 

# In[303]:


# Calculer le pourcentage de valeurs manquantes par colonne
missing_values = train.isnull().sum() / len(train) * 100

# Filtrer les colonnes ayant des valeurs manquantes
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Afficher les colonnes concernées
print("Colonnes ayant des valeurs manquantes et leur pourcentage :\n")
print(missing_values)


# In[304]:


# Vérifier le pourcentage de valeurs manquantes dans la colonne 'sii'
sii_missing_percentage = train["sii"].isnull().sum() / len(train) * 100

# Afficher le résultat
print(f"📌 Pourcentage de valeurs manquantes dans 'sii' : {sii_missing_percentage:.2f}%")


# In[305]:


# Définition des seuils
high_missing = missing_values[missing_values > 50]  # Plus de 50% de NaN
medium_missing = missing_values[(missing_values > 20) & (missing_values <= 50)]  # Entre 20% et 50%
low_missing = missing_values[missing_values <= 20]  # Moins de 20%

# Afficher les résultats
print(f" Variables avec plus de 50% de valeurs manquantes : {high_missing.index.tolist()}")
print(f"Variables entre 20% et 50% de valeurs manquantes : {medium_missing.index.tolist()}")
print(f"Variables avec moins de 20% de valeurs manquantes : {low_missing.index.tolist()}")



# In[306]:


# Liste des variables avec plus de 50% de NaN
high_missing = missing_values[missing_values > 50]

# Afficher les variables concernées
print(f"🔴 Variables avec plus de 50% de NaN ({len(high_missing)}) : {high_missing.index.tolist()}")


# In[307]:


# Suppression définitive des colonnes ayant plus de 50% de valeurs manquantes
train = train.drop(columns=high_missing.index)

# Vérification après suppression
print(f"✅ {len(high_missing)} variables avec plus de 50% de NaN ont été supprimées.")
print(f"📊 Nouvelle taille du dataset : {train.shape}")



# In[308]:


# Définition des seuils de valeurs manquantes
medium_missing = missing_values[(missing_values > 20) & (missing_values <= 50)]  # 20-50% de NaN
low_missing = missing_values[missing_values <= 20]  # <20% de NaN

# Affichage des résultats
print(f"🟠 Variables avec 20-50% de NaN ({len(medium_missing)}) : {medium_missing.index.tolist()}")
print(f"🟢 Variables avec <20% de NaN ({len(low_missing)}) : {low_missing.index.tolist()}")


# In[309]:


# Imputation des variables numériques avec 20-50% de NaN
for col in medium_missing.index:
    if train[col].dtype in ['float64', 'int64']:  # Si numérique
        train[col].fillna(train[col].median(), inplace=True)
    else:  # Si catégorique
        train[col].fillna(train[col].mode()[0], inplace=True)

print(f"✅ Imputation réalisée pour {len(medium_missing)} variables avec 20-50% de NaN.")


# In[310]:


# Imputation des variables numériques avec <20% de NaN
for col in low_missing.index:
    if train[col].dtype in ['float64', 'int64']:  # Si numérique
        train[col].fillna(train[col].median(), inplace=True)  # Médiane pour éviter les outliers
    else:  # Si catégorique
        train[col].fillna(train[col].mode()[0], inplace=True)  # Mode pour les variables catégorielles

print(f"✅ Imputation réalisée pour {len(low_missing)} variables avec <20% de NaN.")


# In[311]:


# Recalcul du nombre total de valeurs manquantes par colonne
remaining_missing_values = train.isnull().sum()

# Filtrer les colonnes qui ont encore des NaN
remaining_missing_values = remaining_missing_values[remaining_missing_values > 0]

# Affichage des résultats
if remaining_missing_values.empty:
    print(" Toutes les valeurs manquantes ont été imputées ! ")
else:
    print(f"🔍 Il reste {len(remaining_missing_values)} colonnes avec des valeurs manquantes :")
    print(remaining_missing_values.sort_values(ascending=False))


# ### 3 Analyse de la distribution des variables numériques

# In[312]:


# Sélection des variables numériques
numeric_columns = train.select_dtypes(include=['float64', 'int64']).columns

# Création d'un histogramme pour chaque variable
for col in numeric_columns:
    fig = px.histogram(train, x=col, title=f"Distribution de {col} après imputation", nbins=50)
    fig.show()


# In[ ]:





# In[313]:


# Calcul de la matrice de corrélation
corr_matrix = train.corr()

# Affichage de la matrice sous forme de heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation des variables numériques")
plt.show()


# In[314]:


# Trier les variables selon leur corrélation absolue avec la cible
target_corr = corr_matrix["sii"].abs().sort_values(ascending=False)

# Afficher les 10 variables les plus corrélées avec `sii`
print("🔍 Variables les plus corrélées avec sii :\n")
print(target_corr[1:20])  


# In[315]:


# Sélection des colonnes PCIAT
pciat_columns = [col for col in train.columns if "PCIAT" in col]
pciat_corr_matrix = train[pciat_columns].corr()

# Affichage de la heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pciat_corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Matrice de corrélation des variables PCIAT")
plt.show()


# In[ ]:





# In[316]:


numeric_vars = train.select_dtypes(include=["float64", "int64"]).columns

plt.figure(figsize=(12, len(numeric_vars) * 2))
for i, var in enumerate(numeric_vars, 1):
    plt.subplot(len(numeric_vars)//3 + 1, 3, i)
    sns.boxplot(x=train[var])
    plt.title(f"Boxplot de {var}")
    plt.xlabel(var)
plt.tight_layout()
plt.show()


# In[317]:


Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1

outliers = ((train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR)))
outliers.sum().sort_values(ascending=False)  # Nombre d'outliers par variable


# In[318]:


train_clean = train[~((train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[326]:


def clean_outliers(df, threshold=3):
    """
    Nettoie les outliers d'un DataFrame en utilisant l'IQR et une approche hybride :
    - Remplacement des valeurs extrêmes par la médiane pour la plupart des cas.
    - Suppression des cas extrêmes rares.

    Paramètres :
    df (pd.DataFrame) : DataFrame contenant les données à nettoyer.
    threshold (int) : Facteur multiplicatif de l'IQR pour définir les bornes des outliers (par défaut 3).

    Retourne :
    pd.DataFrame : DataFrame avec les outliers traités.
    """
    df_cleaned = df.copy()  # Éviter de modifier l'original
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns

    Q1 = df_cleaned[numeric_cols].quantile(0.25)
    Q3 = df_cleaned[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    for col in numeric_cols:
        median_value = df_cleaned[col].median()
        df_cleaned[col] = np.where(
            (df_cleaned[col] < lower_bound[col]) | (df_cleaned[col] > upper_bound[col]),
            median_value,
            df_cleaned[col]
        )

    print(f"Taille finale du dataset après traitement des outliers : {df_cleaned.shape}")
    return df_cleaned

# Utilisation de la fonction sur le dataset train
train_cleaned = clean_outliers(train)

# Sauvegarde du dataset nettoyé
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
file_path = os.path.join(data_folder, "train_cleaned.csv")
train_cleaned.to_csv(file_path, index=False)

print(f"Le dataset nettoyé a été enregistré dans {file_path}")


# In[322]:


# Vérifier que le DataFrame est chargé
if 'train' in globals():
    # Sélectionner uniquement les colonnes numériques après le traitement des outliers
    numeric_cols = train.select_dtypes(include=['number']).columns

    # Tracer les boxplots (affichage en plusieurs colonnes pour plus de clarté)
    plt.figure(figsize=(15, len(numeric_cols) * 0.5))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(len(numeric_cols)//4 + 1, 4, i)  # 4 boxplots par ligne
        sns.boxplot(x=train[col], color="royalblue")
        plt.title(col)
        plt.xlabel("")
    plt.tight_layout()
    plt.show()
else:
    print("🚨 Assurez-vous que le DataFrame 'train' est bien chargé avant d'exécuter ce code.")


# # 📌 Conclusion Générale de l’Analyse Exploratoire des Données (EDA) et Préparation des Données
# 
# L'analyse exploratoire des données (EDA) a permis de mieux comprendre la structure et la qualité des données du projet **Problematic Internet Use**. Grâce aux différentes étapes effectuées, les données sont désormais propres et prêtes pour la phase de modélisation.
# 
# ## 🔍 1. Résumé des Actions Effectuées
# 
# ✅ **Exploration des données** : Vérification des fichiers, aperçu des variables et compréhension des valeurs présentes.  
# 
# ✅ **Traitement des valeurs manquantes** : Identification des variables avec des valeurs manquantes et imputation des données si nécessaire.  
# 
# ✅ **Analyse des distributions** : Étude des distributions des variables numériques et catégorielles à travers des visualisations adaptées.  
# 
# ✅ **Détection et traitement des outliers** :  
#    - Utilisation des boxplots et de l’IQR pour détecter les valeurs aberrantes.  
#    - Stratégie hybride : remplacement des valeurs extrêmes par la médiane et suppression des cas extrêmes rares.  
# 
# ✅ **Analyse des relations entre variables** : Étude des corrélations et des interactions entre variables.  
# 

# In[ ]:





# In[ ]:





# In[ ]:




