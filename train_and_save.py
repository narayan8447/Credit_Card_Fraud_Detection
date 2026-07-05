"""Train and save the hybrid fraud detection model.

The model combines an autoencoder representation of the original numerical
credit-card features with simple contextual features used by the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


def build_autoencoder(input_dim: int = 30, latent_dim: int = 10) -> tuple[Model, Model]:
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(20, activation="relu")(input_layer)
    encoded = Dense(
        latent_dim,
        activation="relu",
        activity_regularizer=regularizers.l1(1e-5),
    )(encoded)

    decoded = Dense(20, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic app-level context features for demonstration purposes."""
    rng = np.random.default_rng(RANDOM_STATE)
    enriched = df.copy()
    enriched["Is_Online"] = rng.integers(0, 2, size=len(enriched))
    enriched["Location_Changed"] = rng.integers(0, 2, size=len(enriched))
    enriched["Multiple_Attempts"] = rng.integers(0, 2, size=len(enriched))
    enriched["Card_Type"] = rng.integers(0, 2, size=len(enriched))
    return enriched


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Download the Kaggle credit card fraud dataset "
            "and place it in the project root before training."
        )

    print("Training hybrid Autoencoder + Random Forest model...")
    df = pd.read_csv(DATA_PATH)
    df = add_context_features(df)

    x_base = df.drop("Class", axis=1)
    y = df["Class"]

    pd.DataFrame([x_base.iloc[:, :30].mean().values]).to_csv(
        MODELS_DIR / "feature_means.csv",
        index=False,
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_base.iloc[:, :30])
    x_context = x_base.iloc[:, 30:].values

    x_normal = x_scaled[y == 0]
    autoencoder, encoder_model = build_autoencoder(input_dim=30, latent_dim=10)

    print("Training autoencoder...")
    autoencoder.fit(
        x_normal,
        x_normal,
        epochs=10,
        batch_size=256,
        shuffle=True,
        verbose=1,
    )

    x_encoded = encoder_model.predict(x_scaled, verbose=0)
    x_hybrid = np.concatenate([x_encoded, x_context], axis=1)

    sampler = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = sampler.fit_resample(x_hybrid, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled,
        y_resampled,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_resampled,
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    random_forest.fit(x_train, y_train)

    y_pred = random_forest.predict(x_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = f"{report}\nAccuracy: {accuracy:.6f}\n"

    print("\nClassification Report:")
    print(report_text)
    (REPORTS_DIR / "training_report.txt").write_text(report_text, encoding="utf-8")

    encoder_model.save(MODELS_DIR / "encoder_model.h5")
    joblib.dump(random_forest, MODELS_DIR / "rf_hybrid_model.joblib", compress=3)
    joblib.dump(scaler, MODELS_DIR / "hybrid_scaler.joblib", compress=3)

    print("Model artifacts saved successfully.")


if __name__ == "__main__":
    main()
