import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import os
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

API_ADRESS = os.getenv("API_ADRESS", default="http://project7api.francecentral.azurecontainer.io:8000")

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

X_train = pd.read_csv("data/X_train.csv").set_index(keys=["SK_ID_CURR"])

st.title("Dashboard prédiction client")

# Asking client ID

df = pd.read_csv("data/data_test.csv").set_index(keys=["SK_ID_CURR"])
list_options = list(df.index)
option = st.selectbox(
    "Pour quelle personne voulez vous voir les résultats ?", (list_options)
)

with st.sidebar:
    st.write(option)
    st.table(df.loc[option, :].T)  # Show the data of the client on sidebar

st.divider()

st.header("Prédiction du crédit")

var_client = df.loc[option, :].rename({"Unnamed: 0": "SK_ID_CURR"})
liste_colonnes_nulles = list(var_client.loc[var_client.isnull()].index)
for colonne in liste_colonnes_nulles:
    var_client[colonne] = dict_imputer[colonne]

client_dict = dict(var_client)
for cle, valeur in client_dict.items():
    if isinstance(valeur, np.int64):
        client_dict[cle] = int(valeur)

client_json = json.dumps(client_dict)


def call_api_prediction(id:str) -> str:
    r = requests.post(f"{API_ADRESS}/model-results", data=id)
    return r.json()


if st.button("Prédire", type="primary"):
    model_result = call_api_prediction(id=client_json)
    st.write(f"{model_result}")


st.divider()

st.header("SHAP client")



df = pd.DataFrame(data=client_dict, index=[0])
df[var_num] = scaler.transform(df[var_num])
df[var_cat] = encoder.transform(df[var_cat])
data = df[colonnes]
shap.initjs()
explainer = shap.TreeExplainer(classifier._final_estimator)
shap_values = explainer.shap_values(data)


st_shap(shap.summary_plot(shap_values, data))

#st_shap(shap.plots.waterfall(explainer.expected_value[0]))

#st_shap(shap.plot_force(explainer.expected_value[0], shap_values[0], data.iloc[0,:]))


st.divider()

st.header("SHAP population")

shap_values_train = shap.TreeExplainer(classifier._final_estimator).shap_values(X_train)
st_shap(shap.summary_plot(shap_values_train, X_train))


list_features = ['EXT_SOURCE_3',
                 'EXT_SOURCE_1',
                 'EXT_SOURCE_2',
                 'AMT_ANNUITY',
                 'AMT_CREDIT',
                 'DAYS_EMPLOYED',
                 'DAYS_BIRTH',
                 'NAME_EDUCATION_TYPE',
                 'DAYS_ID_PUBLISH',
                 'DAYS_REGISTRATION',]

option_feature = st.selectbox(
    "Pour quelle feature voulez vous comparer ?", (list_features)
)


chart_data = X_train[option_feature]
feature_data = pd.Series([data[option_feature].values] * len(X_train[option_feature]))

fig, ax = plt.subplots()
ax.scatter(chart_data.index, chart_data.values, color='red', marker='o', s=10)
ax.plot(chart_data.index, feature_data.values, color='green')
ax.legend()
st.pyplot(fig)


fig, ax = plt.subplots()
ax.hist(chart_data, bins=20)
ax.axvline(data[option_feature].values, color = 'g')
st.pyplot(fig)
#ax.bar(range(len(chart_data)))
#ax.bar(range(len(feature_data)))
#st.bar_chart(chart_data)
#st.bar_chart(data[option_feature])