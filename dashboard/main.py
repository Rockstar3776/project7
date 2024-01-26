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

st.text_input('Movie title', 'Life of Brian')

value_input = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

option = st.selectbox(
    'Pour quelle personne voulez vous voir les résultats ?',
    ('Paul', 'Léa', 'Autres'))

st.write('Vous avez sélectionné:', option)

def call_api(id: str) -> dict:
    r = requests.get('http://127.0.0.1:8000/models-results', params={"id": id})
    return r.json()

model_result = call_api(id=option)
st.write(f"Le résultat du model est: {model_result}")