from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
import json

app = FastAPI()

# Load models
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
zone_encoders = joblib.load('zone_encoders.pkl')
stats_df = pd.read_parquet('ml_model.parquet')
session = ort.InferenceSession('label_model.onnx', providers=['CPUExecutionProvider'])

required_features = [
    'neo_make', 'neo_model', 'neo_engine', 'neo_trim', 'neo_year', 'price', 'miles',
    'Q3_miles', 'Q3_price', 'Q1_price', 'Q1_miles',
    'min_price', 'mean_miles', 'mean_price',
    'max_price', 'min_miles', 'max_miles',
    'price_zone_enc', 'miles_zone_enc',
    'price_to_mean', 'price_diff_from_mean',
    'miles_to_mean', 'miles_diff_from_mean'
]

stat_cols = [
    "min_price", "Q1_price", "mean_price", "Q3_price", "max_price",
    "min_miles", "Q1_miles", "mean_miles", "Q3_miles", "max_miles"
]

class CarItem(BaseModel):
    neo_make: str
    neo_model: str
    neo_year: int
    neo_engine: str = ""
    neo_trim: str = ""
    price: float
    miles: float

def merge_stats_fast(single_df):
    keys = ['neo_make', 'neo_model', 'neo_year', 'neo_engine', 'neo_trim']
    for col in keys:
        single_df[col] = single_df[col].fillna('').astype(str)
        stats_df[col] = stats_df[col].fillna('').astype(str)
    return pd.merge(single_df, stats_df, on=keys, how='left')

def assign_zone_vectorized(df, value_col, prefix):
    conds = [
        df[value_col] < df[f'min_{prefix}'],
        (df[value_col] >= df[f'min_{prefix}']) & (df[value_col] < df[f'Q1_{prefix}']),
        (df[value_col] >= df[f'Q1_{prefix}']) & (df[value_col] < df[f'mean_{prefix}']),
        (df[value_col] >= df[f'mean_{prefix}']) & (df[value_col] <= df[f'Q3_{prefix}']),
        (df[value_col] > df[f'Q3_{prefix}']) & (df[value_col] <= df[f'max_{prefix}']),
        df[value_col] > df[f'max_{prefix}']
    ]
    choices = ['Extreme Low', 'Very Low', 'Low', 'Mid', 'High', 'Very High']
    return np.select(conds, choices, default=None)

def classify_by_matrix(row):
    price = row["price"]
    miles = row["miles"]
    if price == 0 or pd.isnull(price) or pd.isnull(miles):
        return "No Rating"
    if any(pd.isnull(row[col]) for col in stat_cols):
        return "Uncertain"
    return None

@app.post("/predict")
def predict_deal_type(car: CarItem):
    df = pd.DataFrame([car.dict()])
    df = merge_stats_fast(df)

    if df['mean_price'].isna().any():
        return {"deal_type": "Not Found", "reason": "No statistics found for the combination."}

    deal_type_direct = classify_by_matrix(df.iloc[0])
    if deal_type_direct:
        return {"deal_type": deal_type_direct}

    # Encode categorical
    for col in ['neo_make', 'neo_model', 'neo_engine', 'neo_trim']:
        le = label_encoders[col]
        val = str(df[col].values[0])
        if val in le.classes_:
            df[col] = le.transform([val])[0]
        else:
            df[col] = -1

    # Assign zones
    df['price_zone'] = assign_zone_vectorized(df, 'price', 'price')
    df['miles_zone'] = assign_zone_vectorized(df, 'miles', 'miles')

    for col in ['price_zone', 'miles_zone']:
        le = zone_encoders[col]
        val = df[col].values[0]
        if val in le.classes_:
            df[col + '_enc'] = le.transform([val])[0]
        else:
            df[col + '_enc'] = -1

    # Feature engineering
    df['price_to_mean'] = df['price'] / df['mean_price']
    df['price_diff_from_mean'] = df['price'] - df['mean_price']
    df['miles_to_mean'] = df['miles'] / df['mean_miles']
    df['miles_diff_from_mean'] = df['miles'] - df['mean_miles']

    # Fill and scale
    df[required_features] = df[required_features].fillna(0).astype(np.float32)
    X_scaled = scaler.transform(df[required_features])
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_scaled})[0]
    predicted_label = target_encoder.inverse_transform(preds.astype(int))[0]

    # Add extra fields
    result = {
        "deal_type": predicted_label,
        "min_price": float(df['min_price']),
        "max_price": float(df['max_price']),
        "Q1_price": float(df['Q1_price']),
        "Q3_price": float(df['Q3_price']),
        "difference_price": float(df['price'] - df['mean_price']),
        "difference_miles": float(df['miles'] - df['mean_miles']),
        "price_status": "Below Average" if df['price'].values[0] < df['mean_price'].values[0]
                        else "Above Average" if df['price'].values[0] > df['mean_price'].values[0]
                        else "Average",
        "miles_status": "Below Average" if df['miles'].values[0] < df['mean_miles'].values[0]
                        else "Above Average" if df['miles'].values[0] > df['mean_miles'].values[0]
                        else "Average"
    }

    return result
