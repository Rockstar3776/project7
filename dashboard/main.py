import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import os
import joblib
import shap
from streamlit_shap import st_shap

API_ADRESS = os.getenv("API_ADRESS", default="http://127.0.0.1:8000")

model_path = os.path.join("pickle/pipeline_classifier.pkl")
classifier = joblib.load(model_path)
colonnes_path = os.path.join("pickle/dict_imputer.plk")
dict_imputer = joblib.load(colonnes_path)
scaler_path = os.path.join("pickle/pipeline_scaler.pkl")
scaler = joblib.load(scaler_path)
encod_path = os.path.join("pickle/pipeline_encod.pkl")
encoder = joblib.load(encod_path)
var_cat_path = os.path.join("pickle/var_cat.pkl")
var_cat = joblib.load(var_cat_path)
var_num_path = os.path.join("pickle/var_num.pkl")
var_num = joblib.load(var_num_path)
colonnes_path = os.path.join("pickle/colonnes.pkl")
colonnes = joblib.load(colonnes_path)

st.title("Dashboard prédiction client")

# Asking client ID

df = pd.read_csv("data/data_test.csv").set_index(keys=["SK_ID_CURR"])
list_options = list(df.index)
option = st.selectbox(
    "Pour quelle personne voulez vous voir les résultats ?", (list_options)
)


# st.table(df.loc[option, :].T)  # Show the data of the client

st.divider()

var_client = df.loc[option, :].rename({"Unnamed: 0": "SK_ID_CURR"})
liste_colonnes_nulles = list(var_client.loc[var_client.isnull()].index)
for colonne in liste_colonnes_nulles:
    var_client[colonne] = dict_imputer[colonne]

client_dict = dict(var_client)
for cle, valeur in client_dict.items():
    if isinstance(valeur, np.int64):
        client_dict[cle] = int(valeur)

client_json = json.dumps(client_dict)
st.write(client_dict)



def call_api_prediction(id:str) -> str:
    r = requests.post(f"{API_ADRESS}/model-results", data=id)
    return r.json()


if st.button("Prédire", type="primary"):
    model_result = call_api_prediction(id=client_json)
    st.write(f"{model_result}")


def call_api_shapvalues(id:str) -> list:
    r = requests.post(f"{API_ADRESS}/model-shap", data=id)
    print(r)
    return r.json()



df = pd.DataFrame(data=client_dict, index=[0])
df[var_num] = scaler.transform(df[var_num])
df[var_cat] = encoder.transform(df[var_cat])
data = df[colonnes]
shap.initjs()
explainer = shap.TreeExplainer(classifier._final_estimator)
shap_values = np.asarray(explainer.shap_values(data))


st_shap(shap.plots.force(explainer.expected_value[0], shap_values[0,:], data.iloc[0,:]))






# X = pd.read_csv("data/data_test.csv")
# y = pd.read_csv("data/target.csv")
# 
# df_target = pd.concat([X, y], axis=1)
# 
# ligne0 = list(df_target.loc[df_target["TARGET"] == 0].mean())
# ligne1 = list(df_target.loc[df_target["TARGET"] == 1].mean())
# ligne_cli = list(df_target.loc[df_target["SK_ID_CURR"] == option].mean())
# 
# chart_global = (
#     pd.DataFrame([ligne0, ligne1, ligne_cli], columns=df_target.columns)
#     .drop(columns=["SK_ID_CURR", "Unnamed: 0", "TARGET"])
#     .T
# )
# 
# st.bar_chart(chart_global)
# 
# st.divider()
# 
# list_columns = [
#     "EXT_SOURCE_3",
#     "EXT_SOURCE_2",
#     "EXT_SOURCE_1",
#     "CODE_GENDER",
#     "DAYS_EMPLOYED",
#     "NAME_EDUCATION_TYPE",
#     "AMT_CREDIT",
#     "DAYS_BIRTH",
# ]
# 
# option_col = st.selectbox(
#     "Pour quelle catégorie voulez vous voir les résultats ?", (list_columns)
# )
# 
# chart_data = df.reset_index()
# chart_data["colors"] = np.where(chart_data["SK_ID_CURR"] == option, "target", "others")
# 
# st.scatter_chart(chart_data, x="SK_ID_CURR", y=option_col, color="colors")
# 
# 
# st.write("Vous avez sélectionné:", option)
# 
# 
# def call_api(id: str) -> dict:
#     r = requests.get(f"{API_ADRESS}/models-results", params={"id": id})
#     return r.json()
# 
# 
# model_result = call_api(id=option)
# 
# 
# if st.button("Prédire", type="primary"):
#     st.write(f"{model_result}")
# 