import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

st.title('Dashboard prédiction client')

# Asking client ID

df = pd.read_csv("data/data_.csv").set_index(keys=["SK_ID_CURR"])
list_options = list(df.index)
option = st.selectbox(
    'Pour quelle personne voulez vous voir les résultats ?',
    (list_options))


st.table(df.loc[option,:].T)  # Show the data of the client

st.divider()

X = pd.read_csv("data/data_.csv")
y = pd.read_csv("data/target.csv")

df_target = pd.concat([X,y], axis=1)

ligne0 = list(df_target.loc[df_target['TARGET']==0].mean())
ligne1 = list(df_target.loc[df_target['TARGET']==1].mean())
ligne_cli = list(df_target.loc[df_target['SK_ID_CURR']==option].mean())

chart_global = pd.DataFrame([ligne0, ligne1, ligne_cli], 
                            columns=df_target.columns).drop(columns=['SK_ID_CURR','Unnamed: 0', 'TARGET']).T

st.bar_chart(chart_global)

st.divider()

list_columns = ['EXT_SOURCE_3',
                'EXT_SOURCE_2',
                'EXT_SOURCE_1',
                'CODE_GENDER',
                'DAYS_EMPLOYED',
                'NAME_EDUCATION_TYPE',
                'AMT_CREDIT',
                'DAYS_BIRTH']

option_col = st.selectbox(
    'Pour quelle catégorie voulez vous voir les résultats ?',
    (list_columns))

chart_data = df.reset_index()
chart_data['colors'] = np.where(chart_data["SK_ID_CURR"] == option, 'target', 'others')

st.scatter_chart(chart_data, x='SK_ID_CURR', y=option_col, color='colors') 




st.write('Vous avez sélectionné:', option)


def call_api(id: str) -> dict:
    r = requests.get('http://127.0.0.1:8000/models-results', params={"id": id})
    return r.json()
model_result = call_api(id=option)


if st.button("Prédire", type="primary"):
    st.write(f"{model_result}")



