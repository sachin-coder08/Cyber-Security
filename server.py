from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import joblib

from features import extract_features

app = FastAPI()

# For dev: allow localhost:3000. In prod, use your frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    confidence: int
    features: List[float]

MODEL_PATH = "model.joblib"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: URLRequest):
    url = req.url
    features = extract_features(url)  # list of floats

    # If you have a trained model (scikit-learn), use it.
    if model is not None:
        proba = model.predict_proba([features])[0]  # e.g. [prob_not_phish, prob_phish]
        confidence = int(round(100 * proba[1]))    # % probability of phishing
        is_phishing = proba[1] > 0.5
    else:
        # fallback heuristic/dummy
        from random import randint
        confidence = randint(40, 90)
        is_phishing = "login" in url.lower() or "bank" in url.lower()

    return {
        "url": url,
        "is_phishing": bool(is_phishing),
        "confidence": confidence,
        "features": features,
    }
