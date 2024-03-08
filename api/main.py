from fastapi import FastAPI
import os
import joblib
import pandas as pd
from pydantic import BaseModel
import shap

app = FastAPI()

model_path = os.path.join("pickle/pipeline_classifier.pkl")
classifier = joblib.load(model_path)
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



class InputModel(BaseModel):
    SK_ID_CURR : int
    NAME_CONTRACT_TYPE: object
    CODE_GENDER: object
    FLAG_OWN_CAR: object
    FLAG_OWN_REALTY: object
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: object
    NAME_INCOME_TYPE: object
    NAME_EDUCATION_TYPE: object
    NAME_FAMILY_STATUS: object
    NAME_HOUSING_TYPE: object
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: object
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: object
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: object
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    APARTMENTS_AVG: float
    BASEMENTAREA_AVG: float
    YEARS_BEGINEXPLUATATION_AVG: float
    YEARS_BUILD_AVG: float
    COMMONAREA_AVG: float
    ELEVATORS_AVG: float
    ENTRANCES_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LANDAREA_AVG: float
    LIVINGAPARTMENTS_AVG: float
    LIVINGAREA_AVG: float
    NONLIVINGAPARTMENTS_AVG: float
    NONLIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    BASEMENTAREA_MODE: float
    YEARS_BEGINEXPLUATATION_MODE: float
    YEARS_BUILD_MODE: float
    COMMONAREA_MODE: float
    ELEVATORS_MODE: float
    ENTRANCES_MODE: float
    FLOORSMAX_MODE: float
    FLOORSMIN_MODE: float
    LANDAREA_MODE: float
    LIVINGAPARTMENTS_MODE: float
    LIVINGAREA_MODE: float
    NONLIVINGAPARTMENTS_MODE: float
    NONLIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    BASEMENTAREA_MEDI: float
    YEARS_BEGINEXPLUATATION_MEDI: float
    YEARS_BUILD_MEDI: float
    COMMONAREA_MEDI: float
    ELEVATORS_MEDI: float
    ENTRANCES_MEDI: float
    FLOORSMAX_MEDI: float
    FLOORSMIN_MEDI: float
    LANDAREA_MEDI: float
    LIVINGAPARTMENTS_MEDI: float
    LIVINGAREA_MEDI: float
    NONLIVINGAPARTMENTS_MEDI: float
    NONLIVINGAREA_MEDI: float
    FONDKAPREMONT_MODE: object
    HOUSETYPE_MODE: object
    TOTALAREA_MODE: float
    WALLSMATERIAL_MODE: object
    EMERGENCYSTATE_MODE: object
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_WEEK: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float


@app.get("/")
def root():
    return {"message": "Hello, World"}




@app.post("/model-results")
def get_model_results(inputs: InputModel) -> str:
    df = pd.DataFrame(data=[inputs.model_dump()]).set_index(keys=["SK_ID_CURR"])
    df[var_num] = scaler.transform(df[var_num])
    df[var_cat] = encoder.transform(df[var_cat])
    data = df[colonnes]
    resultat = classifier.predict_proba(data)[0]
    print(resultat)
    if resultat[1] > 0.75:
        prediction = "Le client a son prêt"
    else:
        prediction = "Le client n'a pas son prêt"
    
    return prediction


@app.post("/model-shap")
def get_shap(inputs: InputModel) -> list:
    df = pd.DataFrame(data=[inputs.model_dump()]).set_index(keys=["SK_ID_CURR"])
    df[var_num] = scaler.transform(df[var_num])
    df[var_cat] = encoder.transform(df[var_cat])
    data = df[colonnes]
    shap.initjs()
    explainer = shap.TreeExplainer(classifier._final_estimator)
    shap_values = explainer.shap_values(data)
    liste = shap_values[0].tolist() + shap_values[1].tolist() 
    return liste