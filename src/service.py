from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

class ClientData(BaseModel):
    V10: float
    V14: float
    V17: float
    V4: float
    V12: float
    V7: float
    V6: float
    Amount: float

app = FastAPI()

# Загружаем модель
model_path = '../models/xgb_model.pkl'
model = joblib.load(model_path)
print("XGBoost модель загружена")
print(f"Модель ожидает {model.n_features_in_} фичей")

# Порог
THRESHOLD = 0.3

@app.post("/fraud")
def fraud(client_data: ClientData):

    features = [
        client_data.V10,
        client_data.V14, 
        client_data.V17,
        client_data.V4,
        client_data.V12,
        client_data.V7,
        client_data.V6,
        client_data.Amount
    ]
    
    features_array = np.array(features).reshape(1, -1)
    
    probability = model.predict_proba(features_array)[0][1]
    
    prediction = probability >= THRESHOLD
    
    return {
        "fraud": bool(prediction),
        "probability": float(probability),
        "threshold": THRESHOLD
    }

@app.get("/info")
def info():
    if model is None:
        return {"status": "no model loaded"}
    
    return {
        "model": "XGBoost",
        "features": ['V10', 'V14', 'V17', 'V4', 'V12', 'V7', 'V6', 'Amount'],
        "threshold": THRESHOLD,
        "n_features": model.n_features_in_
    }