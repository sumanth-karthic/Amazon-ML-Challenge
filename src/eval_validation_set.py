import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

df = pd.read_parquet('../models/val_predictions.parquet')

print("--- 1-3. Validation Set Metrics ---")
y_true = df['label']
preds = df['pred_prob']

thresholds = [0.98, 0.99, 0.995]
for th in thresholds:
    y_pred = (preds >= th).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    print(f"Threshold {th}:\n  F0.5: {f05:.4f}\n  Precision: {p:.4f}\n  Recall: {r:.4f}\n  False Positives: {fp}\n  False Negatives: {fn}\n")
