
import streamlit as st

import requests
import json

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from flask import Flask


URL_API = "http://localhost:5000/"

def main():

    init = st.markdown("*Initialisation de l'application en cours...*")
    init = st.markdown(init_api())

    # Affichage du titre et du sous-titre
    st.title("Implémenter un modèle de scoring")
    st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)

    # Affichage d'informations dans la sidebar
    st.sidebar.subheader("Informations générales")
    # Chargement du logo
    logo = load_logo()
    st.sidebar.image(logo,
                     width=200)

    # Chargement de la selectbox
    lst_id = load_selectbox()
    global id_client
    id_client = st.sidebar.selectbox("ID Client", lst_id)
    
    # Chargement des infos générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

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
    
    # Affichage de l'ID client sélectionné
    st.write("Vous avez sélectionné le client :", id_client)

    # Affichage état civil
    st.header("**Informations client**")
    #infos = st.checkbox("Afficher les informations du client?")

    if st.checkbox("Afficher les informations du client?"):
        
        infos_client = identite_client()
        #st.write("Statut famille :**", infos_client["status_famille"], "**")
        #st.write("Nombre d'enfant(s) :**", infos_client["nb_enfant"], "**")
        #st.write("Age client :", infos_client["age"], "ans.")
        st.write("Statut famille :**", infos_client["NAME_FAMILY_STATUS"][0], "**")
        st.write("Nombre d'enfant(s) :**", infos_client["CNT_CHILDREN"][0], "**")
        st.write("Age client :", int(infos_client["DAYS_BIRTH"].values / -365), "ans.")

        data_age = load_age_population()
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
        st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

        data_revenus = load_revenus_population()
        # Set the style of plots
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(9, 9))
        # Plot the distribution of revenus
        plt.hist(data_revenus, edgecolor = 'k')
        plt.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="red", linestyle=":")
        plt.title('Revenus du Client')
        plt.xlabel('Revenus ($ USD)')
        plt.ylabel('Count')
        st.pyplot()

        #st.write("Montant du crédit :", infos_client["montant_credit"], "$")
        #st.write("Annuités crédit :", infos_client["annuites"], "$")
        #st.write("Montant du bien pour le crédit :", infos_client["montant_bien"], "$")
        st.write("Montant du crédit :", infos_client["AMT_CREDIT"][0], "$")
        st.write("Annuités crédit :", infos_client["AMT_ANNUITY"][0], "$")
        st.write("Montant du bien pour le crédit :", infos_client["AMT_GOODS_PRICE"][0], "$")
    else:
        st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
    
    # Affichage solvabilité client
    st.header("**Analyse dossier client**")
    
    st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
    prediction = load_prediction()
    st.write(round(prediction*100, 2), "%")
    st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
    st.write(identite_client()) 

    # Affichage des dossiers similaires
    chk_voisins = st.checkbox("Afficher dossiers similaires?")

    if chk_voisins:
        
        similar_id = load_voisins()
        st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
        st.write(similar_id)
        st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)


@st.cache
def init_api():

    # Requête permettant de récupérer la liste des ID clients
    init_api = requests.get(URL_API + "init_model")
    init_api = init_api.json()

    return "Initialisation application terminée."

@st.cache()
def load_logo():
    # Construction de la sidebar
    # Chargement du logo
    logo = Image.open("logo.png") 
    
    return logo

@st.cache()
def load_selectbox():
    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()

    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])

    return lst_id

@st.cache()
def load_infos_gen():

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()

    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]

    # Requête permettant de récupérer
    # Le nombre de target dans la classe 0
    # et la classe 1
    targets = requests.get(URL_API + "disparite_target")    
    targets = targets.json()


    return nb_credits, rev_moy, credits_moy, targets


def identite_client():

    # Requête permettant de récupérer les informations du client sélectionné
    infos_client = requests.get(URL_API + "infos_client", params={"id_client":id_client})
    #infos_client = infos_client.json()
    
    # On transforme la réponse en dictionnaire python
    infos_client = json.loads(infos_client.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    infos_client = pd.DataFrame.from_dict(infos_client).T

    return infos_client

@st.cache
def load_age_population():
    
    # Requête permettant de récupérer les âges de la 
    # population pour le graphique situant le client
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()

    return data_age

@st.cache
def load_revenus_population():
    
    # Requête permettant de récupérer des tranches de revenus 
    # de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")
    
    data_revenus = data_revenus_json.json()

    return data_revenus

def load_prediction():
    
    # Requête permettant de récupérer la prédiction
    # de faillite du client sélectionné
    prediction = requests.get(URL_API + "predict", params={"id_client":id_client})
    prediction = prediction.json()

    return prediction[1]

def load_voisins():
    
    # Requête permettant de récupérer les 10 dossiers
    # les plus proches de l'ID client choisi
    voisins = requests.get(URL_API + "load_voisins", params={"id_client":id_client})

    # On transforme la réponse en dictionnaire python
    voisins = json.loads(voisins.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    voisins = pd.DataFrame.from_dict(voisins).T

    # On déplace la colonne TARGET en premier pour plus de lisibilité
    target = voisins["TARGET"]
    voisins.drop(labels=["TARGET"], axis=1, inplace=True)
    voisins.insert(0, "TARGET", target)
    
    return voisins

if __name__ == "__main__":
    main()