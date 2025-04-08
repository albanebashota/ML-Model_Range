from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
functions_df = pd.read_pickle("label.pkl")
globals_dict = {"pd": pd, "np": np}
for code in functions_df["Code"]:
    exec(code, globals_dict)
# Load helper functions from the pickle
rating_from_value = globals_dict["rating_from_value"]
append_uncertain = globals_dict["append_uncertain"]
combine_final_label = globals_dict["combine_final_label"]
# Initialize FastAPI
app = FastAPI(title="Car Deal Prediction API", description="Predict the deal type of a car based on price and mileage", version="1.0")
# Load car details
stats_df = pd.read_parquet("ml_model.parquet")
# Define input schema
class CarInput(BaseModel):
    neo_make: str
    neo_model: str = None
    neo_engine: str = None
    neo_year: int
    price: float
    miles: float
@app.post("/predict", summary="Predict deal type for a car", response_description="Prediction result with statistics")
def predict_label(car: CarInput):
    df = pd.DataFrame([car.dict()])
    df['neo_model'] = df['neo_model'].astype(str)
    df['neo_engine'] = df['neo_engine'].astype(str)
    print(f"Input data: {df}")  # Debug log
    # Determine merge keys based on availability
    if df['neo_engine'].isna().all() or df['neo_engine'].eq("None").all():
        merge_keys = ['neo_make', 'neo_model', 'neo_year']
    elif df['neo_model'].isna().all() or df['neo_model'].eq("None").all():
        merge_keys = ['neo_make', 'neo_year']
    else:
        merge_keys = ['neo_make', 'neo_model', 'neo_year', 'neo_engine']
    merged = df.merge(stats_df, how='left', on=merge_keys)
    if merged.isnull().any().any():
        print(f"Missing values after merge: {merged[merged.isnull().any(axis=1)]}")
        raise HTTPException(status_code=404, detail="Statistics not found for this car combination.")
    merged['price'] = df['price'].values
    merged['miles'] = df['miles'].values
    row = merged.iloc[0]
    count = row['count']
    # Price and miles rating
    label_price = rating_from_value(
        row['price'], row['min_price'], row['Q1_price'],
        row['mean_price'], row['Q3_price'], row['max_price']
    )
    label_price = append_uncertain(label_price, count)
    label_miles = rating_from_value(
        row['miles'], row['min_miles'], row['Q1_miles'],
        row['mean_miles'], row['Q3_miles'], row['max_miles']
    )
    label_miles = append_uncertain(label_miles, count)
    final_label = combine_final_label(label_price, label_miles, count)
    return {
        "min_price": int(row['min_price']),
        "max_price": int(row['max_price']),
        "Q1_price": int(row['Q1_price']),
        "Q3_price": int(row['Q3_price']),
        "difference_price": int(row['price'] - row['mean_price']),
        "difference_miles": int(row['miles'] - row['mean_miles']),
        "price_status": (
            "Below Average" if row['price'] < row['mean_price']
            else "Above Average" if row['price'] > row['mean_price']
            else "Average"
        ),
        "miles_status": (
            "Below Average" if row['miles'] < row['mean_miles']
            else "Above Average" if row['miles'] > row['mean_miles']
            else "Average"
        ),
        "prediction": final_label
    }
