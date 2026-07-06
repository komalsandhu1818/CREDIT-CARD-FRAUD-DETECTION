"""
generate_transactions.py

Generates a synthetic credit card transaction dataset with:
  - ~0.25% fraud rate (matching real-world datasets like Kaggle CCFD)
  - V1-V20: anonymized PCA-style features (as in real CCFD datasets)
  - Time and Amount features
  - Ground truth label: 0 = legitimate, 1 = fraud

Fraud mix:
  - 92% distinctive fraud: strong feature signal (V1, V4, V14, V17 separated
    by 6-10 sigma from legit distribution)
  - 8% subtle "card testing" fraud: small amounts, overlaps with legit —
    this is what limits precision and makes the problem realistic
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_TRANSACTIONS = 200_000
FRAUD_RATE = 0.0025
OUTPUT_PATH = Path(__file__).parent / "transactions.csv"

N_FRAUD = int(N_TRANSACTIONS * FRAUD_RATE)   # 500
N_LEGIT = N_TRANSACTIONS - N_FRAUD           # 199,500


def generate_legitimate(rng, n):
    feats = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 21)}
    feats["V1"]  = rng.normal( 0.0, 0.55, n)
    feats["V4"]  = rng.normal( 0.0, 0.55, n)
    feats["V14"] = rng.normal( 0.0, 0.55, n)
    feats["V17"] = rng.normal( 0.0, 0.55, n)
    feats["Amount"] = np.clip(np.abs(rng.exponential(60, n)), 0.5, 1500)
    feats["Time"]   = rng.uniform(0, 172800, n)
    feats["Class"]  = 0
    return pd.DataFrame(feats)


def generate_fraud(rng, n):
    n_distinct = int(n * 0.92)
    n_subtle   = n - n_distinct

    # Distinctive fraud: clear PCA feature signature
    Df = {f"V{i}": rng.normal(0, 1, n_distinct) for i in range(1, 21)}
    Df["V1"]  = rng.normal(-7.0,  0.7, n_distinct)
    Df["V4"]  = rng.normal(-6.0,  0.7, n_distinct)
    Df["V14"] = rng.normal(-10.0, 0.8, n_distinct)
    Df["V17"] = rng.normal(-7.0,  0.7, n_distinct)
    Df["V2"]  = rng.normal( 4.5,  0.8, n_distinct)
    Df["Amount"] = rng.exponential(450, n_distinct)
    Df["Time"]   = rng.uniform(0, 172800, n_distinct)
    Df["Class"]  = 1
    distinct = pd.DataFrame(Df)

    # Subtle card-testing fraud: small amounts, overlapping features
    Sf = {f"V{i}": rng.normal(0, 1, n_subtle) for i in range(1, 21)}
    Sf["V1"]  = rng.normal(-1.0, 1.3, n_subtle)
    Sf["V14"] = rng.normal(-1.5, 1.5, n_subtle)
    Sf["Amount"] = rng.uniform(0.5, 25, n_subtle)
    Sf["Time"]   = rng.uniform(0, 172800, n_subtle)
    Sf["Class"]  = 1
    subtle = pd.DataFrame(Sf)

    return pd.concat([distinct, subtle], ignore_index=True)


def main():
    rng = np.random.default_rng(RNG_SEED)
    legit = generate_legitimate(rng, N_LEGIT)
    fraud = generate_fraud(rng, N_FRAUD)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} transactions | "
          f"{df['Class'].sum()} fraud ({df['Class'].mean()*100:.3f}%)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
