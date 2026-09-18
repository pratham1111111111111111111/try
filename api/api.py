from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Traffic Anomaly Detection API",
    description="API for predicting traffic vehicle counts",
    version="1.0.0"
)


# ==========================================
# 2. LOAD MODEL
# ==========================================

MODEL_PATH = "models/traffic_model.joblib"

model = joblib.load(MODEL_PATH)


# ==========================================
# 3. REQUEST DATA FORMAT
# ==========================================

class TrafficRequest(BaseModel):
    Junction: int
    Year: int
    Month: int
    Day: int
    Hour: int
    DayOfWeek: int


# ==========================================
# 4. ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Traffic Anomaly Detection API",
        "status": "running",
        "version": "1.0.0"
    }


# ==========================================
# 5. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: TrafficRequest):

    input_data = pd.DataFrame(
        [
            {
                "Junction": data.Junction,
                "Year": data.Year,
                "Month": data.Month,
                "Day": data.Day,
                "Hour": data.Hour,
                "DayOfWeek": data.DayOfWeek
            }
        ]
    )

    prediction = model.predict(input_data)

    return {
        "predicted_vehicles": float(prediction[0]),
        "junction": data.Junction,
        "status": "prediction successful"
    }


# ==========================================
# 7. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )