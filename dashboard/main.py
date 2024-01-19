import streamlit as st
import requests
import json

st.title('OpenClassrooms Project 7')
st.title('_Léa_ is :blue[awesome] :sunglasses:')

option = st.selectbox(
    'Pour quelle personne voulez vous voir les résultats ?',
    ('Paul', 'Léa', 'Autre'))

st.write('Vous avez sélectionné:', option)

def call_api(id: str) -> dict:
    r = requests.get('http://127.0.0.1:8000/models-results', params={"id": id})
    return r.json()

model_result = call_api(id=option)
st.write(f"Le résultat du model est: {model_result}")