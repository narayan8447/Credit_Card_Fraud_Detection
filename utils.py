"""Model loading and inference helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_FOREST_PATH = MODELS_DIR / "rf_hybrid_model.joblib"
SCALER_PATH = MODELS_DIR / "hybrid_scaler.joblib"
ENCODER_PATH = MODELS_DIR / "encoder_model.h5"
FEATURE_MEANS_PATH = MODELS_DIR / "feature_means.csv"
FRAUD_THRESHOLD = 0.50


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path.name}. "
            "Run train_and_save.py or restore the saved model artifacts before launching the app."
        )


def load_model_and_scaler():
    """Load the trained Random Forest, scaler, and encoder artifacts."""
    for path in (RANDOM_FOREST_PATH, SCALER_PATH, ENCODER_PATH):
        _require_file(path)

    model = joblib.load(RANDOM_FOREST_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = load_model(ENCODER_PATH, compile=False)
    return model, scaler, encoder


def prepare_features(time_seconds, amount, online, location_changed, attempts, card_type):
    """Create the base and contextual feature arrays expected by the hybrid model."""
    _require_file(FEATURE_MEANS_PATH)

    means = pd.read_csv(FEATURE_MEANS_PATH).iloc[0].to_numpy(dtype=float)
    if means.shape[0] != 30:
        raise ValueError(
            f"feature_means.csv must contain 30 base features, found {means.shape[0]}."
        )

    means[0] = float(time_seconds)
    means[-1] = float(amount)

    base = means.reshape(1, -1)
    context = np.array(
        [[online, location_changed, attempts, card_type]],
        dtype=float,
    )
    return base, context


def predict_transaction(model, scaler, encoder, base, context):
    """Return the predicted fraud label and probability for one transaction."""
    scaled = scaler.transform(base)
    encoded = encoder.predict(scaled, verbose=0)
    hybrid = np.concatenate([encoded, context], axis=1)

    probability = float(model.predict_proba(hybrid)[0][1])
    label = "Fraudulent" if probability >= FRAUD_THRESHOLD else "Normal"
    return {"label": label, "probability": round(probability, 3)}
