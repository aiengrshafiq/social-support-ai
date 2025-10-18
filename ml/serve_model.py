# ml/serve_model.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from typing import List

# --- Define Input Schema ---
# Should match the features the model was trained on
class EligibilityFeatures(BaseModel):
    income: float = Field(..., ge=0)
    family_size: int = Field(..., gt=0)
    # Add other features if your dummy model uses them

# --- Define Output Schema ---
class EligibilityPrediction(BaseModel):
    prediction: int # 0 or 1
    probability: float # Probability of class 1 (eligible)
    decision: str # User-friendly decision

# --- Load the Model ---
model_path = os.path.join(os.path.dirname(__file__), 'models', 'eligibility_model_v1.joblib')
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Expected feature order (must match training)
EXPECTED_FEATURES = ['income', 'family_size'] # Adjust if needed

# --- Create FastAPI App ---
app = FastAPI(
    title="Eligibility Prediction Service",
    description="API to predict social support eligibility.",
    version="1.0"
)

@app.post("/predict", response_model=EligibilityPrediction)
async def predict_eligibility(features: EligibilityFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input features to DataFrame in the correct order
        input_data = pd.DataFrame([features.model_dump()], columns=EXPECTED_FEATURES)

        # Make prediction
        prediction_proba = model.predict_proba(input_data)
        prediction = int(model.predict(input_data)[0]) # Get the class prediction (0 or 1)
        probability_eligible = float(prediction_proba[0][1]) # Probability of class 1

        # Determine user-friendly decision based on threshold (e.g., 0.5)
        decision = "Approve" if probability_eligible >= 0.5 else "Decline"
        # Add a 'Review' state for borderline cases later if needed

        return EligibilityPrediction(
            prediction=prediction,
            probability=probability_eligible,
            decision=decision
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# --- Run with Uvicorn (for local testing if needed) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)