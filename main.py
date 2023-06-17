import dill
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open("model/model.pickle", "rb") as file:
    model = dill.load(file)

# Создание переменных для скалирования числовых признаков ("device_screen_width" и "device_screen_height")

df_for_featuring = pd.read_csv("dataframe/df_for_featuring.csv")
df_for_featuring["len"] = df_for_featuring.device_screen_resolution.apply(lambda x : len(x.split("x")))
df_for_featuring.drop(df_for_featuring[df_for_featuring.len == 1].index, inplace=True)
df_for_featuring.drop(columns=["len"], inplace=True)
df_for_featuring["device_screen_width"] = df_for_featuring.device_screen_resolution.apply(lambda x : x.split("x")[0])
df_for_featuring["device_screen_height"] = df_for_featuring.device_screen_resolution.apply(lambda x : x.split("x")[1])
df_for_featuring["device_screen_width"] = df_for_featuring.device_screen_width.astype(int)
df_for_featuring["device_screen_height"] = df_for_featuring.device_screen_height.astype(int)

min_width = min(df_for_featuring["device_screen_width"])
max_width = max(df_for_featuring["device_screen_width"])
min_height = min(df_for_featuring["device_screen_height"])
max_height = max(df_for_featuring["device_screen_height"])

# Создание шаблона для отправки в модель

df_for_modeling = pd.read_csv("dataframe/df_for_modeling.csv")
df_for_modeling.drop(["goal_action", "visit_number", "visit_month", "visit_day", "visit_hour", "visit_minute", "visit_second"],
                     axis=1, inplace=True)
pattern = df_for_modeling.iloc[[0]]
pattern.iloc[[0]] = 0


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    Result: int


@app.get("/status")
def status():
    return "I'm OK. Really..."


@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    data = pd.DataFrame.from_dict([form.dict()])
    data.drop(["device_model"], axis=1, inplace=True)

# Создание стобцов "device_screen_width" и "device_screen_height"

    data["len"] = data.device_screen_resolution.apply(lambda x: len(x.split("x")))
    data.drop(data[data.len == 1].index, inplace=True)
    data.drop(columns=["len"], inplace=True)
    data["device_screen_width"] = data.device_screen_resolution.apply(lambda x: x.split("x")[0])
    data["device_screen_height"] = data.device_screen_resolution.apply(lambda x: x.split("x")[1])
    data["device_screen_width"] = data.device_screen_width.astype(int)
    data["device_screen_height"] = data.device_screen_height.astype(int)
    data.drop(["device_screen_resolution"], axis=1, inplace=True)

# Заполнение трафарета единицами в нужных местах (для категориальных переменных)

    features_for_encode = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
                           "device_category", "device_os", "device_brand", "device_browser", "geo_country", "geo_city"]
    df_to_encode = data[features_for_encode]
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df_to_encode)
    values = ohe.transform(df_to_encode)
    data[ohe.get_feature_names_out()] = values
    data.drop(features_for_encode, axis=1, inplace=True)

    for col in data.columns:
        if col in pattern:
            pattern.loc[0, col] = 1
        else:
            name_list = col.split("_")
            name = f"{name_list[0]}_{name_list[1]}_infrequent_sklearn"
            pattern.loc[0, name] = 1

# Скалирование числовых признаков ("device_screen_width" и "device_screen_height")

    pattern.loc[0, "device_screen_width"] = (data.loc[0, "device_screen_width"] - min_width) / (max_width - min_width)
    pattern.loc[0, "device_screen_height"] = (data.loc[0, "device_screen_height"] - min_height) / (max_height - min_height)

    return {
        "Result": model.predict(pattern)[0]
    }










