"""Shared inference helpers used by the Streamlit app and the notebook."""
from pathlib import Path
import joblib
import pandas as pd
from tensorflow import keras

BASE = Path(__file__).resolve().parent
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

_model = None
_preprocessor = None


def load_artifacts():
    global _model, _preprocessor
    if _model is None:
        _model = keras.models.load_model(BASE / "models" / "model.keras")
        _preprocessor = joblib.load(BASE / "models" / "preprocessor.pkl")
    return _model, _preprocessor


def predict_proba(customer: dict) -> float:
    model, pre = load_artifacts()
    row = pd.DataFrame([customer], columns=FEATURES)
    return float(model.predict(pre.transform(row), verbose=0)[0][0])
