"""
Wine Quality Prediction
=======================
Trains and evaluates three classifier models:
  - Random Forest Classifier
  - Stochastic Gradient Descent (SGD) Classifier
  - Support Vector Classifier (SVC)

Usage:
    python src/train.py
    python src/train.py --data data/WineQT.csv --output models/
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Missing values : {df.isnull().sum().sum()}")
    return df


def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"[INFO] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────

def evaluate(name: str, model, X_tr, X_te, y_tr, y_te, vis_dir: str):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"  Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"{'=' * 55}")
    print(classification_report(y_te, y_pred, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(vis_dir, f"cm_{slug}.png"), dpi=150)
    plt.close()

    return acc


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────

def plot_quality_distribution(df: pd.DataFrame, vis_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["quality"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("muted", len(counts)))
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    ax.set_title("Wine Quality Score Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "quality_distribution.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, vis_dir: str):
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_feature_importances(rf_model, feature_names, vis_dir: str):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values,
            color=sns.color_palette("muted", len(importances)))
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importances — Random Forest", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_importances.png"), dpi=150)
    plt.close()


def plot_model_comparison(results: dict, vis_dir: str):
    res_df = pd.Series(results).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if v == res_df.max() else "#3498db" for v in res_df.values]
    bars = ax.bar(res_df.index, res_df.values * 100, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontweight="bold")
    ax.axhline(y=res_df.max() * 100, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "model_comparison.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(data_path: str, output_dir: str, vis_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load & preprocess
    df = load_data(data_path)
    plot_quality_distribution(df, vis_dir)
    plot_correlation_heatmap(df, vis_dir)

    X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, scaler = preprocess(df)
    feature_names = df.drop(columns=["quality"]).columns

    # Define models
    models = [
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_tr, X_te),
        ("SGD Classifier",           SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),              X_tr_sc, X_te_sc),
        ("SVC (RBF kernel)",         SVC(kernel="rbf", C=10, gamma="scale", random_state=42),              X_tr_sc, X_te_sc),
    ]

    results = {}
    rf_model = None

    for name, model, Xtr, Xte in models:
        acc = evaluate(name, model, Xtr, Xte, y_tr, y_te, vis_dir)
        results[name] = acc
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(output_dir, f"{slug}.pkl"))
        if "random_forest" in slug:
            rf_model = model

    # Additional plots
    if rf_model:
        plot_feature_importances(rf_model, feature_names, vis_dir)
    plot_model_comparison(results, vis_dir)

    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("\n[DONE] All models saved to:", output_dir)
    print("[DONE] Visualisations saved to:", vis_dir)
    best = max(results, key=results.get)
    print(f"[BEST] {best}  →  {results[best] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality Prediction")
    parser.add_argument("--data",   default="data/WineQT.csv",   help="Path to dataset CSV")
    parser.add_argument("--output", default="models/",            help="Directory to save trained models")
    parser.add_argument("--vis",    default="visualizations/",    help="Directory to save plots")
    args = parser.parse_args()

    main(args.data, args.output, args.vis)
