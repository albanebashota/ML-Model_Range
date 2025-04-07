from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
import json
app = FastAPI()
# Ngarko komponentët e modelit
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
stats_df = pd.read_parquet('ml_model.parquet')
session = ort.InferenceSession('label_model.onnx', providers=['CPUExecutionProvider'])
# Lista e input features që kërkohen nga modeli ONNX
required_features = [
    'neo_make', 'neo_model', 'neo_engine', 'neo_year', 'price', 'miles',
    'Q3_miles', 'Q3_price', 'Q1_price', 'Q1_miles',
    'min_price', 'mean_miles', 'mean_price', 'max_price', 'min_miles', 'max_miles'
]
# Formati i input-it për FastAPI
class CarInput(BaseModel):
    neo_make: str
    neo_model: str = None
    neo_engine: str = None
    neo_year: int
    price: float
    miles: float
@app.post("/predict")
def predict_deal_type(car: CarInput):
    # Krijo DataFrame për një veturë
    df = pd.DataFrame([car.dict()])
    # Zëvendëso 0 ose bosh me NA
    df.replace({"neo_engine": {0: None}, "neo_model": {0: None}}, inplace=True)
    # Konverto në string për label encoding
    df['neo_model'] = df['neo_model'].astype("string")
    df['neo_engine'] = df['neo_engine'].astype("string")
    # Zgjidh kolonat për merge sipas të dhënave
    if df['neo_engine'].isna().all():
        merge_cols = ['neo_make', 'neo_model', 'neo_year']
    elif df['neo_model'].isna().all():
        merge_cols = ['neo_make', 'neo_year']
    else:
        merge_cols = ['neo_make', 'neo_model', 'neo_year', 'neo_engine']
    # Bëj merge me statistikat
    merged = df.merge(stats_df, on=merge_cols, how='left')
    if merged.isnull().any().any():
        raise HTTPException(status_code=400, detail="Statistikat mungojnë për këtë veturë.")
    # Rikthe vlerat origjinale për price dhe miles
    merged['price'] = df['price'].values
    merged['miles'] = df['miles'].values
    # Label encoding për kategoritë
    for col in ['neo_make', 'neo_model', 'neo_engine']:
        le = label_encoders[col]
        val = str(merged[col].values[0])
        if val in le.classes_:
            merged[col] = le.transform([val])[0]
        else:
            merged[col] = -1
    # Mbush mungesat me 0
    merged.fillna(0, inplace=True)
    # Inferencë me modelin ONNX
    X = merged[required_features].astype(float)
    X_scaled = scaler.transform(X)
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    label = target_encoder.inverse_transform(pred.astype(int))[0]
    # Përgjigje e plotë JSON
    return {
        "min_price": float(merged['min_price'].values[0]),
        "max_price": float(merged['max_price'].values[0]),
        "Q1_price": float(merged['Q1_price'].values[0]),
        "Q3_price": float(merged['Q3_price'].values[0]),
        "difference_price": float(df['price'].values[0] - merged['mean_price'].values[0]),
        "difference_miles": float(df['miles'].values[0] - merged['mean_miles'].values[0]),
        "price_status": (
            "Below Average" if df['price'].values[0] < merged['mean_price'].values[0]
            else "Above Average" if df['price'].values[0] > merged['mean_price'].values[0]
            else "As Average"
        ),
        "miles_status": (
            "Below Average" if df['miles'].values[0] < merged['mean_miles'].values[0]
            else "Above Average" if df['miles'].values[0] > merged['mean_miles'].values[0]
            else "As Average"
        ),
        "prediction": label
    }
