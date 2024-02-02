from fastapi import FastAPI
import os
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model_path = os.path.join("api/pipeline_model.pkl")
classifier = joblib.load(model_path)

data = pd.read_csv("data/data_.csv").set_index(keys=["SK_ID_CURR"])
target = pd.read_csv("data/target.csv")

class ModelInput(BaseModel):
    NAME_CONTRACT_TYPE: float
    CODE_GENDER: float
    FLAG_OWN_CAR: float
    FLAG_OWN_REALTY: float
    NAME_TYPE_SUITE: float
    NAME_INCOME_TYPE: float
    NAME_EDUCATION_TYPE: float
    NAME_FAMILY_STATUS: float
    NAME_HOUSING_TYPE: float
    OCCUPATION_TYPE: float
    WEEKDAY_APPR_PROCESS_START: float
    ORGANIZATION_TYPE: float
    FONDKAPREMONT_MODE: float
    HOUSETYPE_MODE: float
    WALLSMATERIAL_MODE: float
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    OWN_CAR_AGE: float
    FLAG_WORK_PHONE: float
    FLAG_PHONE: float
    FLAG_EMAIL: float
    REGION_RATING_CLIENT: float
    HOUR_APPR_PROCESS_START: float
    REG_REGION_NOT_LIVE_REGION: float
    REG_REGION_NOT_WORK_REGION: float
    LIVE_REGION_NOT_WORK_REGION: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    LIVE_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    YEARS_BEGINEXPLUATATION_AVG: float
    ELEVATORS_AVG: float
    FLOORSMAX_AVG: float
    TOTALAREA_MODE: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: float
    FLAG_DOCUMENT_5: float
    FLAG_DOCUMENT_6: float
    FLAG_DOCUMENT_8: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_WEEK: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float

test_value = {'NAME_CONTRACT_TYPE': 0.0,
 'CODE_GENDER': 1.0,
 'FLAG_OWN_CAR': 1.0,
 'FLAG_OWN_REALTY': 0.0,
 'NAME_TYPE_SUITE': 6.0,
 'NAME_INCOME_TYPE': 1.0,
 'NAME_EDUCATION_TYPE': 4.0,
 'NAME_FAMILY_STATUS': 1.0,
 'NAME_HOUSING_TYPE': 1.0,
 'OCCUPATION_TYPE': 14.0,
 'WEEKDAY_APPR_PROCESS_START': 4.0,
 'ORGANIZATION_TYPE': 5.0,
 'FONDKAPREMONT_MODE': 2.0,
 'HOUSETYPE_MODE': 0.0,
 'WALLSMATERIAL_MODE': 4.0,
 'CNT_CHILDREN': 2.0,
 'AMT_INCOME_TOTAL': 0.66,
 'AMT_CREDIT': -0.08924812,
 'AMT_ANNUITY': 1.5348606,
 'REGION_POPULATION_RELATIVE': -0.4941845,
 'DAYS_BIRTH': 0.33769092,
 'DAYS_EMPLOYED': 0.1829219,
 'DAYS_REGISTRATION': 0.70719445,
 'DAYS_ID_PUBLISH': -0.40829778,
 'OWN_CAR_AGE': 10.0,
 'FLAG_WORK_PHONE': 0.0,
 'FLAG_PHONE': 0.0,
 'FLAG_EMAIL': 0.0,
 'REGION_RATING_CLIENT': 0.0,
 'HOUR_APPR_PROCESS_START': -0.25,
 'REG_REGION_NOT_LIVE_REGION': 0.0,
 'REG_REGION_NOT_WORK_REGION': 0.0,
 'LIVE_REGION_NOT_WORK_REGION': 0.0,
 'REG_CITY_NOT_LIVE_CITY': 0.0,
 'REG_CITY_NOT_WORK_CITY': 1.0,
 'LIVE_CITY_NOT_WORK_CITY': 1.0,
 'EXT_SOURCE_1': 0.16980723,
 'EXT_SOURCE_2': 0.14388001,
 'EXT_SOURCE_3': -2.438697,
 'YEARS_BEGINEXPLUATATION_AVG': 0.0,
 'ELEVATORS_AVG': 0.0,
 'FLOORSMAX_AVG': 0.0,
 'TOTALAREA_MODE': 0.0,
 'OBS_30_CNT_SOCIAL_CIRCLE': 0.0,
 'DEF_30_CNT_SOCIAL_CIRCLE': 0.0,
 'DEF_60_CNT_SOCIAL_CIRCLE': 0.0,
 'DAYS_LAST_PHONE_CHANGE': 0.58256173,
 'FLAG_DOCUMENT_3': 0.0,
 'FLAG_DOCUMENT_5': 0.0,
 'FLAG_DOCUMENT_6': 0.0,
 'FLAG_DOCUMENT_8': 0.0,
 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0,
 'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0,
 'AMT_REQ_CREDIT_BUREAU_MON': 1.0,
 'AMT_REQ_CREDIT_BUREAU_QRT': 0.0,
 'AMT_REQ_CREDIT_BUREAU_YEAR': 0.0}

@app.get('/')
def root():
    return {'message': 'Hello, World'}


def compute_model_result(parameters: ModelInput):
    df = pd.DataFrame(data=[parameters.model_dump()])
    return classifier.predict_proba(df)[0]



@app.get("/models-results")
def get_models_results(id: int) -> str:
    selected_row = data.loc[id,:]
    value = dict(selected_row)
    model_input = ModelInput.model_validate(value)
    resultat = compute_model_result(parameters=model_input)
    print(resultat)
    if resultat[1]>0.75 :
        prediction = 'Le client a son prêt'
    else :
        prediction = "Le client n'a pas son prêt"

    return prediction