"""
visualize.py

Generates 3 plots for the project README:
  1. Precision-Recall curve for both models
  2. Feature importance bar chart (Random Forest)
  3. Transaction amount distribution: fraud vs legitimate
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

BASE      = Path(__file__).parent.parent
DATA_PATH = BASE / "data"   / "transactions.csv"
MODEL_DIR = BASE / "models"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 21)] + ["Time", "Amount"]


def load_test_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["Class"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    return scaler.transform(X_test), y_test, df


def plot_precision_recall_curve(X_test_s, y_test):
    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    lr = joblib.load(MODEL_DIR / "logistic_regression.joblib")

    rf_proba = rf.predict_proba(X_test_s)[:, 1]
    lr_proba = lr.predict_proba(X_test_s)[:, 1]

    rf_p, rf_r, _ = precision_recall_curve(y_test, rf_proba)
    lr_p, lr_r, _ = precision_recall_curve(y_test, lr_proba)
    rf_auc = average_precision_score(y_test, rf_proba)
    lr_auc = average_precision_score(y_test, lr_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rf_r, rf_p, color="#4C72B0", lw=2,
            label=f"Random Forest (AUC-PR={rf_auc:.4f})")
    ax.plot(lr_r, lr_p, color="#DD8452", lw=2, linestyle="--",
            label=f"Logistic Regression (AUC-PR={lr_auc:.4f})")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Credit Card Fraud Detection")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    out = PLOTS_DIR / "precision_recall_curve.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_feature_importance():
    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    top = imp.sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if "V" in f else "#4C72B0" for f in top.index]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Top Features for Fraud Detection (Random Forest)")
    fig.tight_layout()
    out = PLOTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_amount_distribution(df):
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(legit.clip(upper=500), bins=60, alpha=0.6,
            label=f"Legitimate (n={len(legit):,})", color="#4C72B0", density=True)
    ax.hist(fraud.clip(upper=500), bins=30, alpha=0.6,
            label=f"Fraud (n={len(fraud):,})", color="#C44E52", density=True)
    ax.set_xlabel("Transaction Amount ($, clipped at $500)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount: Legitimate vs Fraud")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "amount_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    X_test_s, y_test, df = load_test_data()
    plot_precision_recall_curve(X_test_s, y_test)
    plot_feature_importance()
    plot_amount_distribution(df)
