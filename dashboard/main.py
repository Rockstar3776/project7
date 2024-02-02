import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

st.title('Dashboard prédiction client')

# Asking client ID

df = pd.read_csv("data/data.csv").set_index(keys=["SK_ID_CURR"])
list_options = list(df.index)
option = st.selectbox(
    'Pour quelle personne voulez vous voir les résultats ?',
    (list_options))


st.table(df.loc[option,:].T)  # Show the data of the client

st.divider()

#chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#st.line_chart(chart_data)


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

chart_data = df[option_col]
colors = ['red' if idx == option else 'blue' for idx in chart_data.index]

st.bar_chart(chart_data, c=colors)




st.write('Vous avez sélectionné:', option)

def call_api(id: str) -> dict:
    r = requests.get('http://127.0.0.1:8000/models-results', params={"id": id})
    return r.json()

model_result = call_api(id=option)
st.write(f"Le résultat du model est: {model_result}")