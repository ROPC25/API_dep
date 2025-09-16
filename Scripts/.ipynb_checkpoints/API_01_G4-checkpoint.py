import os
import pandas as pd
from flask import Flask, jsonify
import joblib
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

app = Flask(__name__)

PATH=chemin_parent1

# Charger le fichier CSV
def load_df(PATH):
    
    csv_path = os.path.join(PATH, 'Simulations','Data_processed','data_test_scaled.csv')
    
    try:
        df = pd.read_csv(csv_path)
        print('df shape =', df.shape)
    except FileNotFoundError:
        df = None
    
    return csv_path,df
    
csv_path,df=load_df(PATH=PATH)


# Chargement du modèle
def load_model(PATH):
    
    model_path = os.path.join(PATH, 'Simulations','Best_model','BestModel.pkl')
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = None
    return model_path,model

model_path,model=load_model(PATH=PATH)
load_clf=model


# Premiers pas sur l'API
@app.route('/')
def index():
    return 'Welcome to my Flask API!'

# Récupération du Client-ID
@app.route('/credit/<id_client>', methods=["GET"])
def credit(id_client):
    print('id client = ', id_client)

# Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]

# Isolement des features non utilisées
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]
    X = X[relevant_features]
    print('X shape = ', X.shape)

# Calcule des valeurs SHAP pour l'échantillon donné
    import shap
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    
    explainer = shap.TreeExplainer(load_clf)
    shap_val = explainer.shap_values(X)

# Prédiction
    proba = load_clf.predict_proba(X)
    prediction = load_clf.predict(X)
    pred_proba = {
        #'prediction': int(prediction),
        'prediction': int(prediction[0]),
        'proba': float(proba[0][0]),
        'shap_values': shap_val.tolist(),
        'X_features': {
        'columns': X.columns.tolist(),
        'values': X.values.tolist()
        }
    }

    #print('Nouvelle Prédiction : \n', pred_proba)

    return jsonify(pred_proba)


# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)