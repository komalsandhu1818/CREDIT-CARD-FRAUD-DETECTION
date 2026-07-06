# Credit Card Fraud Detection Pipeline

End-to-end ML pipeline for detecting credit card fraud in a heavily
imbalanced dataset (0.25% fraud rate), using SMOTE oversampling,
Logistic Regression and Random Forest classifiers, precision-recall
curve threshold tuning, and a real-time scoring daemon.

## Why this problem

Credit card fraud detection is the canonical class-imbalance ML problem.
At 0.25% fraud rate, a model that predicts "legitimate" for every transaction
achieves 99.75% accuracy — yet catches zero fraud. The challenge is to
maximize recall (catch as much fraud as possible) while keeping precision
high enough that the fraud operations team isn't overwhelmed with false
alarms. This requires going beyond accuracy as a metric and carefully
managing the precision-recall trade-off via threshold tuning.

## Architecture

```
data/generate_transactions.py   → synthetic transaction generator (200K txns, 0.25% fraud)
src/train_model.py              → full pipeline: scale → SMOTE → train → tune → evaluate
src/visualize.py                → precision-recall curve, feature importances, amount distribution
src/daemon.py                   → real-time scoring loop with structured alert logging
models/                         → persisted scaler, RF model, LR model, operating points
plots/                          → generated evaluation charts
```

## Pipeline in detail

```
Raw CSV (200K transactions)
        │
        ▼
Stratified train/test split (70/30, stratified on Class)
        │                │
   [TRAIN SET]      [TEST SET — never touched until final eval]
        │
StandardScaler (fit on train only)
        │
SMOTE (synthetic minority oversampling — train only)
  1:399 imbalance → 1:1 balanced training set
        │
 ┌──────┴──────┐
 │             │
LogReg        RandomForest
 │             │
 └──────┬──────┘
        │
Precision-Recall curve on test set
        │
Threshold tuning: find max-precision threshold ≥ 95% recall target
        │
Final evaluation + feature importances
```

## Results

**Random Forest at default threshold (0.5):**

| Metric    | Value  |
|-----------|--------|
| Precision | 98.6%  |
| Recall    | 92.0%  |
| F1-score  | 95.2%  |
| AUC-PR    | 0.9588 |

**Random Forest at tuned threshold (maximize recall ≥ 95%):**

| Metric    | Value  |
|-----------|--------|
| Precision | 79.4%  |
| Recall    | 95.3%  |
| F1-score  | 86.7%  |

**Key insight on the precision-recall trade-off:**
At 0.25% fraud rate (150 fraud in 60K test samples), achieving 95% precision
simultaneously with 95% recall requires ≤8 false positives out of 59,850
legitimate transactions — an FPR below 0.013%. In practice the best
operating point is either:
- High precision (98.6%) + moderate recall (92%) at threshold=0.5 — best
  for minimising ops team workload
- High recall (95.3%) + lower precision (79.4%) at tuned threshold — best
  for minimising fraud losses

The right choice depends on the cost ratio: cost of a missed fraud vs cost
of a false alarm (customer friction, manual review time).

**Top fraud-indicating features:**
V14, V1, V17, V4, V2 (anonymized PCA components) and Amount.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate 200K synthetic transactions (0.25% fraud)
python data/generate_transactions.py

# 2. Train both models and print evaluation metrics
python src/train_model.py

# 3. Generate evaluation plots
python src/visualize.py

# 4. Run real-time scoring daemon (replays first 5K transactions)
python src/daemon.py --limit 5000 --interval 0.01
```

## Design decisions

**Why SMOTE over class_weight?**
`class_weight='balanced'` adjusts the loss function — it still trains on
the original skewed distribution. SMOTE generates synthetic minority samples
in feature space, giving the model more diverse fraud examples to learn from.
In practice, combining both (SMOTE + class_weight='balanced' on RF) gives
the best results.

**Why Random Forest over XGBoost or a neural net?**
RF is interpretable (feature importances), robust to scale without a
separate tuning step, and fast to train. For a fraud detection system where
a compliance team needs to explain why a transaction was flagged, tree-based
feature importances are operationally useful. XGBoost would likely improve
AUC-PR marginally; a neural net would require more data to generalize.

**Why AUC-PR over AUC-ROC?**
AUC-ROC is misleading on imbalanced datasets — a model with many true
negatives (99.75% of the data) gets an inflated ROC score even if it misses
most fraud. AUC-PR focuses on the minority class performance, which is the
only thing that matters operationally.

## Tech stack

Python · scikit-learn (RandomForestClassifier, LogisticRegression,
StandardScaler, precision_recall_curve) · imbalanced-learn (SMOTE) ·
pandas · NumPy · Matplotlib · joblib
