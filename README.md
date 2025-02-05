# Downloading and Using the Machine Learning Model
The API automatically downloads the necessary files from Google Drive if they are not present locally. The files include:

label_encoders.pkl: Encodes categories for neo_make, neo_model, etc.

labels_model.pkl: The main model for predictions.

model.pkl: Reference for ml_model (statistics).

scaler.pkl: Normalizes the input data.

target_encoder.pkl: Converts classes into final labels.

**Endpoint for Predicting a Vehicle's Category**

Method: POST

URL: http://127.0.0.1:8000/docs

**JSON Request:**

{
  "neo_make": "Toyota",
  "neo_model": "Corolla",
  "neo_year": 2020,
  "neo_engine": "1.8L",
  "price": 20000,
  "miles": 30000
}

**JSON Response:**

{
  "prediction": "Fair Deal"
}


**Endpoint for Range**
Method: POST

URL: http://127.0.0.1:8080/docs

