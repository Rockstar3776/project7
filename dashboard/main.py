import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

st.title('OpenClassrooms Project 7')
st.title('_Léa_ is :blue[awesome] :sunglasses:')


df = pd.read_csv("data/data.csv").set_index(keys=["SK_ID_CURR"])

st.dataframe(df)  # Same as st.write(df)

st.divider()

value_imput = st.text_input('ID client', value='', key=int)
st.write("L'ID du client est le :", value_imput)

option = st.selectbox(
    'Pour quelle personne voulez vous voir les résultats ?',
    ('Paul', 'Léa', 'Autres'))

st.write('Vous avez sélectionné:', option)

def call_api(id: str) -> dict:
    r = requests.get('http://127.0.0.1:8000/models-results', params={"id": id})
    return r.json()

model_result = call_api(id=value_imput)
st.write(f"Le résultat du model est: {model_result}")