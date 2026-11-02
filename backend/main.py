from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

bundle = joblib.load("model.joblib")
model = bundle["model"]
COLUMNS = bundle["columns"]

app = FastAPI(title="Diabetes Risk Estimator")

# "*" is fine for a demo. After deploying, replace it with your Vercel URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Person(BaseModel):
    # Ranges match what the model saw in training.
    age: float = Field(ge=1, le=80)
    gender: Literal["Male", "Female"]
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["never", "former", "current", "unknown"]
    bmi: float = Field(ge=10, le=60)
    hba1c: float = Field(ge=3.5, le=9.0)
    glucose: int = Field(ge=80, le=300)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Person):
    row = {
        "age": p.age,
        "hypertension": p.hypertension,
        "heart_disease": p.heart_disease,
        "bmi": p.bmi,
        "HbA1c_level": p.hba1c,
        "blood_glucose_level": p.glucose,
        "gender_Male": 1 if p.gender == "Male" else 0,
        "smoking_history_former": 1 if p.smoking_history == "former" else 0,
        "smoking_history_never": 1 if p.smoking_history == "never" else 0,
        "smoking_history_unknown": 1 if p.smoking_history == "unknown" else 0,
    }
    X = pd.DataFrame([row])[COLUMNS].astype(float)
    flagged = int(model.predict(X)[0])
    return {
        "risk": "higher" if flagged == 1 else "lower",
        "note": "Educational estimate only. Not a diagnosis.",
    }