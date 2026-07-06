"""
daemon.py

Lightweight real-time fraud scoring daemon.

Replays held-out transactions from the dataset one by one, scores each
with the trained Random Forest model, and raises an alert for any
transaction whose fraud probability exceeds the operating threshold.

In a production deployment this would consume from a Kafka topic or a
payment gateway webhook rather than replaying a CSV.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / "models"
ALERT_LOG = BASE / "fraud_alerts.log"

FEATURE_COLS = [f"V{i}" for i in range(1, 21)] + ["Time", "Amount"]


class FraudDaemon:
    def __init__(self):
        self.scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        self.model  = joblib.load(MODEL_DIR / "random_forest.joblib")
        op = joblib.load(MODEL_DIR / "operating_points.joblib")
        self.threshold = op["rf_threshold"]

    def score(self, features: dict) -> tuple[float, bool]:
        x = np.array([[features[c] for c in FEATURE_COLS]])
        x_s = self.scaler.transform(x)
        prob = self.model.predict_proba(x_s)[0, 1]
        return prob, prob >= self.threshold

    def alert(self, tx_id: int, prob: float, amount: float, true_label: int):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tx_id":     tx_id,
            "fraud_prob": round(float(prob), 4),
            "amount":     round(amount, 2),
            "true_label": true_label,
            "correct":    true_label == 1,
        }
        tag = "✓ TRUE FRAUD" if true_label == 1 else "✗ FALSE ALARM"
        print(f"[ALERT] tx={tx_id:>7} | prob={prob:.4f} | "
              f"amount=${amount:>8.2f} | {tag}")
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run(self, source_csv: Path, limit: int, interval: float):
        df = pd.read_csv(source_csv)
        if limit:
            df = df.head(limit)

        n_alerts = n_true = n_false = 0
        print(f"FraudDaemon | threshold={self.threshold:.4f} | "
              f"scanning {len(df):,} transactions\n")

        for idx, row in df.iterrows():
            features = {c: row[c] for c in FEATURE_COLS}
            prob, flagged = self.score(features)
            if flagged:
                self.alert(idx, prob, row["Amount"], int(row["Class"]))
                n_alerts += 1
                if row["Class"] == 1:
                    n_true += 1
                else:
                    n_false += 1
            time.sleep(interval)

        print(f"\nDone. {n_alerts} alerts raised — "
              f"{n_true} true fraud, {n_false} false alarms.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(BASE / "data" / "transactions.csv"))
    parser.add_argument("--limit",    type=int,   default=5000)
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args()

    daemon = FraudDaemon()
    daemon.run(Path(args.source), args.limit, args.interval)


if __name__ == "__main__":
    main()
