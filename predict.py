"""
Wine Quality Prediction — Inference Script
==========================================
Load a saved model and predict wine quality for new samples.

Usage:
    python src/predict.py --model models/random_forest_classifier.pkl --scaler models/scaler.pkl --input sample.csv
    python src/predict.py --model models/svc_rbf_kernel.pkl          --scaler models/scaler.pkl --input sample.csv
"""

import argparse
import os

import joblib
import pandas as pd

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]

SCALED_MODELS = {"sgd_classifier", "svc_rbf_kernel"}


def load_artifacts(model_path: str, scaler_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model_path: str, scaler_path: str, input_path: str):
    model, scaler = load_artifacts(model_path, scaler_path)

    df = pd.read_csv(input_path)
    # Drop Id if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    # Keep only feature columns
    df = df[[f for f in FEATURES if f in df.columns]]

    # Apply scaling for models that need it
    slug = os.path.basename(model_path).replace(".pkl", "")
    if slug in SCALED_MODELS:
        X = scaler.transform(df)
    else:
        X = df.values

    preds = model.predict(X)
    df["predicted_quality"] = preds
    print(df[["predicted_quality"]].to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality — Prediction")
    parser.add_argument("--model",   required=True, help="Path to saved model .pkl")
    parser.add_argument("--scaler",  default="models/scaler.pkl", help="Path to scaler .pkl")
    parser.add_argument("--input",   required=True, help="CSV file with wine features")
    args = parser.parse_args()

    predict(args.model, args.scaler, args.input)
