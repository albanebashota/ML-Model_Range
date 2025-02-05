Deal_ML-Model

# Shkarkimi i Modeleve te Machine Learning
Si modeli ashtu edhe scaler-i jane te ruajtur ne Google Drive.API automatikisht i shkarkon skedaret nga Drive.
Lista e skedareve qe shkarkohen:
1. label_encoders.pkl	Encode kategorite per neo_make, neo_model, etj.
2. labels_model.pkl	Modeli kryesor per parashikime
3. model.pkl	Reference per ml_model(statistikat)
4. scaler.pkl	Normalizimi i te dhenave hyrese
5. target_encoder.pkl	Konvertimi i klasave ne etiketat perfundimtare

# Endpoint për parashikimin e një veture
Metoda: POST

URL: http://127.0.0.1:8000/docs 

 **Shembull i kerkeses JSON**
 
{
  "neo_make": "Toyota",
  
  "neo_model": "Corolla",
  
  "neo_year": 2020,
  "neo_engine": "1.8L",
  "price": 20000,
  "miles": 30000
}


**Pergjigja JSON**


{
  "prediction": "Fair Deal"
}
