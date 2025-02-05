from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from pydantic import BaseModel

# ID-te e skedareve nga Google Drive
file_links = {
    "label_encoders.pkl": "1cfH0qL4zNNwGj-Dm_7jNYmIvli_Obn4U",
    "labels_model.pkl": "1A9yzSITMTtMcI1mWQH52Xr-LlrPx9O2D",
    "model.pkl": "1DFn9xYkVw6pzG2RQ966wGvffF6qF1Qqb",
    "scaler.pkl": "17WvXfDscBh4bY4R00pkPHMjXWbM6Apnw",
    "target_encoder.pkl": "1YIyH817cXA6MZBK-yXR_5_SymqDQEfVa"
}
# Shkarko skedaret nese mungojne
for file_name, file_id in file_links.items():
    if not os.path.exists(file_name):
        print(f"Shkarkimi i {file_name} nga Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)

# Ngarko modelin dhe scaler-in
model = joblib.load("labels_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
scaler = joblib.load("scaler.pkl")
ml_data = joblib.load("model.pkl")

#Struktura e te dhenave hyrese
class CarInput(BaseModel):
    neo_make: str
    neo_model: str
    neo_year: int
    neo_engine: str = ""  
    price: float
    miles: float

app = FastAPI()

@app.post("/predict")
async def predict_category(car: CarInput):
    new_car = car.dict()

    use_engine_filter = new_car["neo_engine"] not in ["", 0, np.nan]

    if new_car["price"] == 0 or np.isnan(new_car["price"]):
        return {"prediction": "No Rating"}

    if use_engine_filter:
        filtered_df = ml_data[
            (ml_data["neo_make"] == new_car["neo_make"]) &
            (ml_data["neo_model"] == new_car["neo_model"]) &
            (ml_data["neo_year"] == new_car["neo_year"]) &
            (ml_data["neo_engine"] == new_car["neo_engine"])
        ]
    else:
        filtered_df = ml_data[
            (ml_data["neo_make"] == new_car["neo_make"]) &
            (ml_data["neo_model"] == new_car["neo_model"]) &
            (ml_data["neo_year"] == new_car["neo_year"])
        ]

    if filtered_df.empty:
        return {"prediction": "Uncertain"}

    new_car["Q3_miles"] = filtered_df["Q3_miles"].mean()
    new_car["Upper_Bound_price"] = filtered_df["Upper_Bound_price"].mean()
    new_car["Q3_price"] = filtered_df["Q3_price"].mean()
    new_car["Q1_price"] = filtered_df["Q1_price"].mean()
    new_car["Q1_miles"] = filtered_df["Q1_miles"].mean()
    new_car["min_price"] = filtered_df["min_price"].mean()
    new_car["max_price"] = filtered_df["max_price"].mean()
    new_car["mean_miles"] = filtered_df["mean_miles"].mean()
    new_car["mean_price"] = filtered_df["mean_price"].mean()

    new_car_df = pd.DataFrame([new_car])
    
    for col in ["neo_make", "neo_model", "neo_engine"]:
        if col in new_car and new_car[col] in label_encoders[col].classes_:
            new_car_df[col] = label_encoders[col].transform([new_car[col]])
        else:
            new_car_df[col] = -1
    
    feature_names = scaler.feature_names_in_
    new_car_df = new_car_df.reindex(columns=feature_names)
    new_car_scaled = scaler.transform(new_car_df)
    
    predicted_label = model.predict(new_car_scaled)
    predicted_category = target_encoder.inverse_transform(predicted_label)[0]
    return {
        "prediction": predicted_category,
        "min_price": new_car["min_price"],
        "max_price": new_car["max_price"],
        "Q1_price": new_car["Q1_price"],
        "Q3_price": new_car["Q3_price"],
        "difference_price": new_car["price"] - new_car["mean_price"],
        "difference_miles": new_car["miles"] - new_car["mean_miles"],
        "price_status": "Above Average" if new_car["price"] > new_car["mean_price"] else "Below Average",
        "miles_status": "High Mileage" if new_car["miles"] > new_car["mean_miles"] else "Low Mileage"
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)