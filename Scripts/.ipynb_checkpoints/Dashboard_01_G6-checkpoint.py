#!/usr/bin/env python
# coding: utf-8

import os
import json
import warnings
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import joblib

warnings.filterwarnings("ignore")

# ========== CHARGEMENT DES DONNÉES ==========
from pathlib import Path

#chemin_courant = Path().resolve()
#chemin_parent = chemin_courant.parent

chemin_fichier = Path(__file__).resolve()         # Chemin complet du fichier .py
#chemin_parent = chemin_fichier.parent            # Chemin dossier contenant le fichier
chemin_parent0= chemin_fichier.parents[0]
chemin_parent1= chemin_fichier.parents[1]
print('chemin_fichier',chemin_fichier)
print('chemin_parent0',chemin_parent0)
print('chemin_parent1',chemin_parent1)

df = pd.read_csv(chemin_parent1/'Simulations'/'Data_processed'/'data_test_scaled.csv') # data_test_scaled (features traitées sans la cible)
data_test = pd.read_csv(chemin_parent1/'Simulations'/'Data_original'/'application_train.csv')  # Infos client
data_train = pd.read_csv(chemin_parent1/'Simulations'/'Data_original'/'application_train.csv')  # Pour comparaison
description = pd.read_csv(chemin_parent1/'Simulations'/'Data_original'/'HomeCredit_columns_description.csv',
                          usecols=['Row', 'Description'],
                          index_col=0,
                          encoding='unicode_escape')

ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df.columns if col not in ignore_features]

#model = joblib.load(os.path.join("Simulations", "Best_model", "model_lgbm.pkl"))
model = joblib.load(os.path.join(chemin_parent1,'Simulations','Best_model',"BestModel.pkl"))

# ========== FONCTIONS UTILITAIRES ==========
@st.cache_data
def get_client_info(data, id_client):
    return data[data['SK_ID_CURR'] == int(id_client)]

@st.cache_data
def get_credit_decision(classe_predite):
    if classe_predite == 1:
        return "Crédit Refusé"
    else:
        return "Crédit Accordé"
        

@st.cache_data
def plot_distribution(applicationDF, feature, client_feature_val, title):
    if pd.isna(client_feature_val):
        st.warning("Valeur manquante pour ce client (NaN). Impossible de comparer.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    t0 = applicationDF[applicationDF['TARGET'] == 0]
    t1 = applicationDF[applicationDF['TARGET'] == 1]

    if feature == "DAYS_BIRTH":
        sns.kdeplot((t0[feature] / -365).dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot((t1[feature] / -365).dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val / -365), color="blue", linestyle='--', label='Client')
        ax.set_xlabel("Âge (années)")
    elif feature == "DAYS_EMPLOYED":
        sns.kdeplot((t0[feature] / 365).dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot((t1[feature] / 365).dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val / 365), color="blue", linestyle='--', label='Client')
        ax.set_xlabel("Ancienneté emploi (années)")
    else:
        sns.kdeplot(t0[feature].dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot(t1[feature].dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val), color="blue", linestyle='--', label='Client')
        ax.set_xlabel(feature)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.legend()
    st.pyplot(fig)

@st.cache_data
def univariate_categorical(applicationDF, feature, client_feature_val, title,
                           ylog=False, label_rotation=False, horizontal_layout=True):
    if pd.isna(client_feature_val.iloc[0]):
        st.warning("Valeur manquante pour ce client (NaN). Impossible de comparer.")
        return

    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Nombre': temp.values})

    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc["TARGET"] *= 100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    layout = (2, 1) if not horizontal_layout else (1, 2)
    fig, (ax1, ax2) = plt.subplots(*layout, figsize=(12, 5) if horizontal_layout else (15, 10))
    fig.subplots_adjust(hspace=0.7) # espace vertical entre figures
    #fig.subplots_adjust(wspace=0.4) # espace horizontal entre figures

    sns.countplot(ax=ax1, x=feature, data=applicationDF,
                  hue="TARGET", order=cat_perc[feature],
                  palette=['g', 'r'])
    pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
    ax1.set_title(title, fontsize=16)
    ax1.axvline(pos1, color="blue", linestyle='--', label='Client')
    if ylog:
        ax1.set_yscale('log')
    if label_rotation:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    sns.barplot(ax=ax2, x=feature, y='TARGET',
                order=cat_perc[feature], data=cat_perc, palette='Set2')
    #sns.barplot(ax=ax2, x=feature, y='TARGET', data=cat_perc, color="skyblue")
    pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
    ax2.axvline(pos2, color="blue", linestyle='--', label='Client')
    if label_rotation:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_title(f"{title} (% Défaillants)", fontsize=16)

    st.pyplot(fig)


# ========== SIDEBAR ==========
import base64

# Charger l’image en base64
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convertir l’image
img_base64 = get_image_base64(chemin_parent0/"logo_entreprise.png")

# Injecter dans HTML
st.sidebar.markdown(
    f"""
    <div style="margin-top: -40px; text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

#st.sidebar.image("logo_entreprise.png", width=300)  # Remplace par ton logo ici

st.sidebar.title("Prêt à dépenser")
st.sidebar.markdown("**Analyse crédit - Dashboard**")

# Seuil de solvabilité à saiisr depuis la sidebar
seuil_solvabilite_str = st.sidebar.text_input(
    "Entrer le seuil de solvabilité [0,1]",
    value="0.5",
    help="Appuyez sur Entrée pour valider"
)

# Conversion et validation du seuil
try:
    seuil_solvabilite = float(seuil_solvabilite_str)
    if not (0.0 <= seuil_solvabilite <= 1.0):
        st.sidebar.error("La valeur doit être comprise entre 0.00 et 1.00")
        seuil_solvabilite = 0.5  # Valeur par défaut
except ValueError:
    st.sidebar.error("Veuillez entrer un nombre valide.")
    seuil_solvabilite = 0.5  # Valeur par défaut

    
#seuil_solvabilite = st.sidebar.number_input(
#    "Entrer le seuil de solvabilité", 
#    min_value=0.0, 
#    max_value=1.0, 
#    value=0.5,     # Valeur par défaut
#    step=0.01, 
#    format="%.2f"
#)

id_list = df["SK_ID_CURR"].values
id_client = st.sidebar.selectbox("Sélectionner ID client", id_list)

with st.sidebar.expander("Options d'affichage", expanded=True):
    show_credit_decision = st.checkbox("Afficher la décision du modèle")
    show_client_details = st.checkbox("Afficher les infos client")
    show_client_comparison = st.checkbox("Comparer au reste des clients")
    shap_general = st.checkbox("Importance globale des variables")

with st.sidebar.expander("Aide sur les variables"):
    feature = st.selectbox("Choisir une variable", sorted(description.index.unique()))
    desc = description.loc[feature, 'Description']
    st.markdown(f"**{desc}**")

# ========== HEADER PRINCIPAL ==========
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg, #0a4275, #3b8ed0);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .subheader {
        font-weight: 600;
        color: #0a4275;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    <div class="header">
        <h1>Dashboard de Scoring Crédit</h1>
        <p>Outil d'aide à la décision pour l’octroi de crédit à destination des gestionnaires</p>
    </div>
    """, unsafe_allow_html=True)

# ========== INFOS CLIENT ==========
client_info = get_client_info(data_test, id_client)
st.markdown(f"### Informations client - ID : {id_client}")

cols_map = {
    'CODE_GENDER': "GENRE", 'DAYS_BIRTH': "AGE", 'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
    'CNT_CHILDREN': "ENFANTS", 'FLAG_OWN_CAR': "VOITURE", 'FLAG_OWN_REALTY': "IMMOBILIER",
    'NAME_EDUCATION_TYPE': "ÉDUCATION", 'OCCUPATION_TYPE': "EMPLOI",
    'DAYS_EMPLOYED': "ANCIENNETÉ", 'AMT_INCOME_TOTAL': "REVENUS", 'AMT_CREDIT': "CRÉDIT",
    'NAME_CONTRACT_TYPE': "CONTRAT", 'AMT_ANNUITY': "ANNUITÉS", 'NAME_INCOME_TYPE': "TYPE REVENU",
    'EXT_SOURCE_1': "EXT1", 'EXT_SOURCE_2': "EXT2", 'EXT_SOURCE_3': "EXT3"
}

df_client = client_info[list(cols_map.keys())].rename(columns=cols_map)
df_client["AGE"] = (-df_client["AGE"] / 365).astype(int)
df_client["ANCIENNETÉ"] = (-df_client["ANCIENNETÉ"] / 365).astype(int)

if show_client_details:
    selected_cols = st.multiselect("Sélectionnez les infos à afficher :", options=df_client.columns,
                                   default=["GENRE", "AGE", "STATUT FAMILIAL", "REVENUS", "CRÉDIT"])
    st.table(df_client[selected_cols].T)

    if st.checkbox("Afficher toutes les données brutes du client"):
        st.dataframe(client_info)

# ========== DÉCISION DU MODÈLE ==========
if show_credit_decision:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Décision du modèle")

    try:
        API_url = f"http://127.0.0.1:5000/credit/{id_client}"
        with st.spinner("Chargement des résultats du modèle..."):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            #prediction = API_data['prediction']
            proba = 1 - API_data['proba']
            # Prédiction binaire selon le seuil personnalisé
            prediction = int(proba > seuil_solvabilite)
            score = round(proba * 100, 2)

            col1, col2 = st.columns([1, 2])
            col1.metric("Risque de défaut", f"{score} %")

            #decision_text = "Crédit Accordé" if prediction == 0 else "Crédit Refusé"
            decision_text = get_credit_decision(classe_predite=prediction)
            color = "green" if prediction == 0 else "red"
            col1.markdown(f"<h3 style='color:{color}; font-weight:bold'>{decision_text}</h3>", unsafe_allow_html=True)

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "lightyellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': score}
                }
            ))
            col2.plotly_chart(gauge_fig, use_container_width=True)

        if st.checkbox("Voir les variables influençant la décision"):
            shap.initjs()
            X = df[df["SK_ID_CURR"] == int(id_client)][relevant_features]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Gestion des types shap_values (liste ou matrice)
            if isinstance(shap_values, list):
                values_to_plot = shap_values[0]
            else:
                values_to_plot = shap_values
            
            # Utiliser un background sample pour explainer (obligatoire pour API moderne)
            background = df[relevant_features]#.sample(10000, random_state=42)
        
            # Utiliser la nouvelle API shap.Explainer
            explainer = shap.Explainer(model, background)
            shap_values = explainer(X, check_additivity=False)  # shap_values devient un objet Explanation
            
            plt.clf()
            plt.close('all')
            plt.figure(figsize=(10, 10))  # Crée une figure que SHAP utilisera

            #fig, ax = plt.subplots(figsize=(10, 8))
            #shap.plots.waterfall(shap_values[0]) 
            #shap.summary_plot(values_to_plot, X, plot_type="bar", show=False)
            #shap.force_plot(explainer.expected_value, shap_values, X.iloc[0], show=False)
            #shap.waterfall_plot(shap_values)
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X.iloc[0], show=False)
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], client_features, show=False)
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            fig = plt.gcf() 
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

# ========== COMPARAISON CLIENT ==========
if show_client_comparison:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Comparaison avec les autres clients")

    selected_var = st.selectbox("Variable à comparer", list(cols_map.values()))
    feature = [k for k, v in cols_map.items() if v == selected_var][0]

    numerical = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    rotate = ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE','NAME_INCOME_TYPE','OCCUPATION_TYPE']
    horizontal = ['OCCUPATION_TYPE', 'NAME_INCOME_TYPE']

    if feature in numerical:
        plot_distribution(data_train, feature, client_info[feature].values[0], selected_var)
    else:
        rotate_lbl = feature in rotate
        layout = feature in horizontal
        univariate_categorical(data_train, feature, client_info[feature], selected_var, False, rotate_lbl, layout)

# ========== IMPORTANCE GLOBALE ==========
if shap_general:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Importance globale des variables")
    st.image('global_feature_importance.png', use_container_width=True)
