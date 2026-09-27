import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

print("Loading train_features.parquet...")
df = pd.read_parquet('dataset/train_features.parquet')

# Merge with val predictions to get just the validation set and its predictions
print("Loading val_predictions.parquet...")
val_preds = pd.read_parquet('models/val_predictions.parquet')

df_val = df.merge(val_preds[['s1_entity_id', 'cand_entity_id', 'pred_prob']], on=['s1_entity_id', 'cand_entity_id'], how='inner')
print(f"Validation set size: {len(df_val):,} pairs")

# Calculate ranks
df_val['rank'] = df_val.groupby('s1_entity_id')['pred_prob'].rank(method='first', ascending=False)
top2 = df_val[df_val['rank'] <= 2].pivot(index='s1_entity_id', columns='rank', values='pred_prob')
top2.columns = ['prob_1', 'prob_2']
top2['prob_2'] = top2['prob_2'].fillna(0)
top2['margin'] = top2['prob_1'] - top2['prob_2']
df_val = df_val.merge(top2[['margin']], on='s1_entity_id', how='left')

def evaluate(y_true, y_pred, name):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    print(f"{name:35s} | F0.5: {f05:.4f} | P: {p:.4f} | R: {r:.4f} | TP: {tp:5d} | FP: {fp:4d} | FN: {fn:4d}")
    return f05, p, r, fp, fn

print("\n--- 9. Targeted Penalty Search ---")
best_f05 = 0.9672 # Base 0.98

for t_base in [0.96, 0.97, 0.975, 0.98]:
    for t_penalty in [0.98, 0.985, 0.99, 0.995]:
        if t_penalty <= t_base: continue
        
        # Rule: Use t_base normally, but t_penalty if address is missing
        pred_addr = (df_val['pred_prob'] >= t_base) & (df_val['address_missing'] == 0)
        pred_miss = (df_val['pred_prob'] >= t_penalty) & (df_val['address_missing'] == 1)
        pred = pred_addr | pred_miss
        evaluate(df_val['label'], pred, f"{t_base} | {t_penalty} if addr_miss")

        # Rule: Use t_base normally, but t_penalty if NOT exact name
        pred_exact = (df_val['pred_prob'] >= t_base) & (df_val['name_exact_match'] == 1)
        pred_non_exact = (df_val['pred_prob'] >= t_penalty) & (df_val['name_exact_match'] == 0)
        pred = pred_exact | pred_non_exact
        evaluate(df_val['label'], pred, f"{t_base} | {t_penalty} if not exact_name")

print("\n--- 10. Margin Rule Search ---")
for t_base in [0.95, 0.96, 0.97, 0.975, 0.98]:
    for margin_req in [0.05, 0.1, 0.2]:
        # Accept if >= t_base AND margin >= margin_req
        # OR accept if >= 0.985 (high confidence override)
        pred = (df_val['pred_prob'] >= 0.985) | ((df_val['pred_prob'] >= t_base) & (df_val['margin'] >= margin_req))
        evaluate(df_val['label'], pred, f"0.985 OR ({t_base} & margin>={margin_req})")

        # Accept if >= t_base AND margin >= margin_req
        # OR accept if >= 0.99 (high confidence override)
        pred = (df_val['pred_prob'] >= 0.99) | ((df_val['pred_prob'] >= t_base) & (df_val['margin'] >= margin_req))
        evaluate(df_val['label'], pred, f"0.99 OR ({t_base} & margin>={margin_req})")
