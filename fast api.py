from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort

app = FastAPI()
.
# Ngarko modelet dhe encoders
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
stats_df = pd.read_parquet('ml_model.parquet')
session = ort.InferenceSession('label_model.onnx', providers=['CPUExecutionProvider'])

class CarInput(BaseModel):
    neo_make: str
    neo_model: str
    neo_year: int
    price: float
    miles: float
    neo_engine: str | None = None  # Optional

@app.post("/predict")
def predict_deal(data: CarInput):
    df = pd.DataFrame([data.dict()])
    df['neo_engine'] = df['neo_engine'].replace({None: pd.NA})
    df['neo_engine'] = df['neo_engine'].astype("string")
    df['neo_model'] = df['neo_model'].astype("string")

    if df['neo_engine'].isna().all():
        merge_cols = ['neo_make', 'neo_model', 'neo_year']
    elif df['neo_model'].isna().all():
        merge_cols = ['neo_make', 'neo_year']
    else:
        merge_cols = ['neo_make', 'neo_model', 'neo_year', 'neo_engine']

    merged = df.merge(stats_df, how='left', on=merge_cols)

    merged['price'] = df['price']
    merged['miles'] = df['miles']

    for col in ['neo_make', 'neo_model', 'neo_engine']:
        le = label_encoders[col]
        merged[col] = merged[col].astype(str)
        merged[col] = merged[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    merged.fillna(0, inplace=True)

    required_features = ['neo_make', 'neo_model', 'neo_engine', 'neo_year', 'price', 'miles']
    X = merged[required_features].astype(float)
    X_scaled = scaler.transform(X)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    deal_type = target_encoder.inverse_transform(preds.astype(int))[0]

    diff_price = float(merged['price'] - merged['mean_price'])
    diff_miles = float(merged['miles'] - merged['mean_miles'])
    price_status = "Below Average" if diff_price < 0 else "Above Average" if diff_price > 0 else "Average"
    miles_status = "Low Mileage" if diff_miles < 0 else "High Mileage" if diff_miles > 0 else "Average"

    return {
        "prediction": deal_type,
        "min_price": float(merged['min_price']),
        "max_price": float(merged['max_price']),
        "Q1_price": float(merged['Q1_price']),
        "Q3_price": float(merged['Q3_price']),
        "difference_price": diff_price,
        "difference_miles": diff_miles,
        "price_status": price_status,
        "miles_status": miles_status
    }
