# Car Deal Prediction API

This FastAPI service predicts the **deal type** of a used car based on various attributes, primarily **price** and **mileage**.
Predictions are made using a **pre-built ML model (`label.pkl`)** also statistical benchmarks (such as percentiles, averages, min/max values) generated from millions of car listings. The service applies rule-based logic on top of these statistics to classify a car as a **Good Deal, Fair Deal, Great Deal**, etc.

Features Used

The model focuses on:

- **Key input features**:  
  - `price`  
  - `miles`

- **Additional features used during XGBoost model training** include:  
  - `neo_make`, `neo_model`, `neo_year`, `neo_engine`  
  - Market-based statistical features (e.g. `Q1_price`, `Q3_price`, `mean_price`, etc.)  
  - Derived ratios and price/mileage deviation metrics

These features are stored in the `ml_model.parquet` and used during inference to ensure accurate classification.

---

Run Locally

To run the service locally, install the required dependencies and launch the FastAPI server:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
Once the server is running, you can access the interactive API documentation (Swagger UI) by opening the following URL in your browser:

http://localhost:8000/docs (or http://127.0.0.1:8000/docs) ```

Example Request
**Endpoint: POST http://localhost:8000/predict
{
  "neo_make": "Audi",
  "neo_model": "A8",
  "neo_engine": "3.0L V6",
  "neo_year": 2021,
  "price": 39278,
  "miles": 33124
}

**Example Response
{
  "min_price": 30650,
  "max_price": 67848,
  "Q1_price": 38791,
  "Q3_price": 45988,
  "difference_price": -2837,
  "difference_miles": -8832,
  "price_status": "Below Average",
  "miles_status": "Below Average",
  "prediction": "GOOD DEAL"
}
