import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

import xgboost as xgb

from joblib import load

@st.cache
def load_data():

    # On charge les données
    data_train = pd.read_csv("app_train.csv")
    data_train.drop("Unnamed: 0", axis=1, inplace=True)
    data_test = pd.read_csv("app_test.csv")
    data_test.drop("Unnamed: 0", axis=1, inplace=True)
    data_train_prepared = pd.read_csv("app_train_prepared.csv")
    data_train_prepared.drop("Unnamed: 0", axis=1, inplace=True)
    data_test_prepared = pd.read_csv("app_test_prepared.csv")
    data_test_prepared.drop("Unnamed: 0", axis=1, inplace=True)

    return data_train, data_test, data_train_prepared, data_test_prepared

@st.cache
def load_xgboost():

    clf_xgb = load("xgboost.pickle")

    return clf_xgb


@st.cache(allow_output_mutation=True)
def load_knn(df_train):

    knn = entrainement_knn(df_train)
    print("Training knn done")

    return knn

#@st.cache()
#def load_logo():
    # Construction de la sidebar
    # Chargement du logo
#    logo = Image.open("logo.png") 
    
#    return logo

@st.cache()
def load_infos_gen(data_train):

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data_train["TARGET"].value_counts()

    return nb_credits, rev_moy, credits_moy, targets


def identite_client(data_test, id):

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    print(data_client)
    print(data_client.columns)

    return data_client

@st.cache
def load_age_population(data_train):
    
    data_age = round((data_train["DAYS_BIRTH"] / -365), 2)

    return data_age

@st.cache
def load_revenus_population(data_train):
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    data_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return data_revenus

def load_prediction(data_test, test, id, clf):
    
    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])
     
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])
    print(test)
   
    data_client = test.iloc[index[0]]

    print(data_client)

    prediction = clf.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return prediction[1]

def load_voisins(data_train, data_test, test, id, mdl):
    
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    index = index[0]
    print(index)

    data_client = pd.DataFrame(test.iloc[index]).T

    print("Analyse :")
    print("Shape data_client :", data_client.shape)
    print(data_client)

    print("Recherche dossiers en cours...")
    distances, indices = mdl.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :].copy()
    
    return df_voisins

def entrainement_knn(df):

    print("Entrainement knn en cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn 


# AFFICHAGE DASHBOARD

init = st.markdown("*Initialisation de l'application en cours...*")
# On charge les données et on initialise l'application
data_train, data_test, data_train_prepared, data_test_prepared = load_data()
id_client = data_test["SK_ID_CURR"].values
clf_xgb = load_xgboost()

init = st.markdown("*Initialisation de l'application terminée...*")

#######################################
# SIDEBAR
#######################################
# Affichage du titre et du sous-titre
st.title("Implémenter un modèle de scoring")
st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)

# Texte de présentation
st.sidebar.header("**PRET A DEPENSER**")

st.sidebar.subheader("Sélection ID_client")

# Chargement de la selectbox
chk_id = st.sidebar.selectbox("ID Client", id_client)


# Affichage d'informations dans la sidebar
st.sidebar.subheader("Informations générales")
# Chargement du logo
# Lors du déploiment sur Azure, l'affichage de l'image mettait le code en erreur.
# Car l'application n'arrivait pas à trouver le fichier.
# J'ai donc enlever cette partie pour le déploiement et l'ai remplacé par du texte.
#logo = load_logo()
#st.sidebar.image(logo,
#                    width=200)

# Chargement des infos générales
nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data_train)

# Affichage des infos dans la sidebar
# Nombre de crédits existants
st.sidebar.markdown("<u>Nombre crédits existants dans la base :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Graphique camembert
st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)

plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%',
        shadow=True, startangle=90)
st.sidebar.pyplot()

# Revenus moyens
st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# Montant crédits moyen
st.sidebar.markdown("<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

#######################################
# PAGE PRINCIPALE
#######################################
# Affichage de l'ID client sélectionné
st.write("Vous avez sélectionné le client :", chk_id)

# Affichage état civil
st.header("**Informations client**")

if st.checkbox("Afficher les informations du client?"):
    
    infos_client = identite_client(data_test, chk_id)
    print(infos_client)
    st.write("Statut famille :**", infos_client["NAME_FAMILY_STATUS"].values[0], "**")
    st.write("Nombre d'enfant(s) :**", infos_client["CNT_CHILDREN"].values[0], "**")
    st.write("Age client :", int(infos_client["DAYS_BIRTH"] / -365), "ans.")

    data_age = load_age_population(data_train)
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9, 9))
    # Plot the distribution of ages in years
    plt.hist(data_age, edgecolor = 'k', bins = 25)
    plt.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle=":")
    plt.title('Age of Client')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    st.pyplot()

    st.subheader("*Revenus*")
    #st.write("Total revenus client :", infos_client["revenus"], "$")
    st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"].values[0], "$")

    data_revenus = load_revenus_population(data_train)
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9, 9))
    # Plot the distribution of revenus
    plt.hist(data_revenus, edgecolor = 'k')
    plt.axvline(infos_client["AMT_INCOME_TOTAL"].values[0], color="red", linestyle=":")
    plt.title('Revenus du Client')
    plt.xlabel('Revenus ($ USD)')
    plt.ylabel('Count')
    st.pyplot()

    #st.write("Montant du crédit :", infos_client["montant_credit"], "$")
    #st.write("Annuités crédit :", infos_client["annuites"], "$")
    #st.write("Montant du bien pour le crédit :", infos_client["montant_bien"], "$")
    st.write("Montant du crédit :", infos_client["AMT_CREDIT"].values[0], "$")
    st.write("Annuités crédit :", infos_client["AMT_ANNUITY"].values[0], "$")
    st.write("Montant du bien pour le crédit :", infos_client["AMT_GOODS_PRICE"].values[0], "$")
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

# Affichage solvabilité client
st.header("**Analyse dossier client**")

st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
prediction = load_prediction(data_test, data_test_prepared, chk_id, clf_xgb)
st.write(round(prediction*100, 2), "%")

st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
st.write(identite_client(data_test, chk_id)) 

# Affichage des dossiers similaires
chk_voisins = st.checkbox("Afficher dossiers similaires?")

if chk_voisins:
    
    knn = load_knn(data_train_prepared)
    st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
    st.dataframe(load_voisins(data_train, data_test, data_test_prepared, chk_id, knn))
    st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
