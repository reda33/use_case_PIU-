#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import os


# In[21]:


# Charger les données nettoyées
train = pd.read_csv("../data/train_cleaned.csv")  # Assurez-vous que le fichier existe

# Vérifier les premières lignes
display(train.head())


# In[22]:


# Création de nouvelles variables pertinentes
# Exemple : Transformation de la variable âge en catégories
def age_category(age):
    if age < 10:
        return "Enfant"
    elif age < 15:
        return "Adolescent"
    else:
        return "Jeune Adulte"

train["Age_Categorie"] = train["Basic_Demos-Age"].apply(age_category)

#Création d'un indicateur de temps d'écran élevé
train["High_Internet_Use"] = (train["PreInt_EduHx-computerinternet_hoursday"] > 2).astype(int)


# In[23]:


# Création d'un indicateur de temps d'écran élevé
train["High_Internet_Use"] = (train["PreInt_EduHx-computerinternet_hoursday"] > 2).astype(int)


# In[24]:


#  Encodage de toutes les variables catégorielles
categorical_cols = train.select_dtypes(include=["object"]).columns
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)


# In[25]:


# Normalisation des variables numériques
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_cols = train.select_dtypes(include=["number"]).columns
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])

# Vérification après transformation
display(train.head())

# Définir le chemin du dossier et du fichier de sauvegarde
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
file_path = os.path.join(data_folder, "train_ready.csv")

# Sauvegarde du dataset prêt pour la modélisation
train.to_csv(file_path, index=False)

print(f"Le dataset prêt a été enregistré dans {file_path}")


# In[ ]:





# In[ ]:




