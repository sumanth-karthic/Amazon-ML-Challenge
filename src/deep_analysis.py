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

def evaluate(y_true, y_pred, name):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    print(f"{name:30s} | F0.5: {f05:.4f} | P: {p:.4f} | R: {r:.4f} | TP: {tp:5d} | FP: {fp:4d} | FN: {fn:4d}")
    return f05, p, r, fp, fn

print("\n--- 1. Base Thresholds ---")
evaluate(df_val['label'], (df_val['pred_prob'] >= 0.98).astype(int), "Base 0.98")
evaluate(df_val['label'], (df_val['pred_prob'] >= 0.99).astype(int), "Base 0.99")
evaluate(df_val['label'], (df_val['pred_prob'] >= 0.995).astype(int), "Base 0.995")

print("\n--- 2. Score Distributions ---")
tp_mask = (df_val['label'] == 1) & (df_val['pred_prob'] >= 0.98)
fp_mask = (df_val['label'] == 0) & (df_val['pred_prob'] >= 0.98)
fn_mask = (df_val['label'] == 1) & (df_val['pred_prob'] < 0.98)

print("TP pred_prob quantiles:", df_val[tp_mask]['pred_prob'].quantile([0.1, 0.5, 0.9]).to_dict())
print("FP pred_prob quantiles:", df_val[fp_mask]['pred_prob'].quantile([0.1, 0.5, 0.9]).to_dict())
print("FN pred_prob quantiles:", df_val[fn_mask]['pred_prob'].quantile([0.1, 0.5, 0.9]).to_dict())

print("\n--- 3. FP Error Analysis ---")
fps = df_val[fp_mask]
print(f"Total FPs (0.98): {len(fps)}")
print("FPs with name_exact_match=1:", fps['name_exact_match'].sum())
print("FPs with missing address:", fps['address_missing'].sum())
print("FPs with name_edit_sim >= 0.9:", (fps['name_edit_sim'] >= 0.9).sum())
print("FPs with addr_edit_sim >= 0.9:", (fps['addr_edit_sim'] >= 0.9).sum())
print("FPs from Source 2:", fps['cand_entity_id'].str.startswith('S2').sum())
print("FPs from Source 3:", fps['cand_entity_id'].str.startswith('S3').sum())

# Cross-source disagreement: entities with matches in both S2 and S3, but maybe high score is only for one?
# For now, let's explore decision rules

print("\n--- 4. Exploring Decision Rules ---")
# Strategy A: Baseline 0.98
base_pred = df_val['pred_prob'] >= 0.98

# Strategy B: High threshold (0.99) unless exact_name
strat_b = (df_val['pred_prob'] >= 0.99) | ((df_val['pred_prob'] >= 0.98) & (df_val['name_exact_match'] == 1))
evaluate(df_val['label'], strat_b, "0.99 OR (0.98 & exact_name)")

# Strategy C: 0.99 unless strong name + strong address
strong_both = (df_val['name_edit_sim'] >= 0.9) & (df_val['addr_edit_sim'] >= 0.9)
strat_c = (df_val['pred_prob'] >= 0.99) | ((df_val['pred_prob'] >= 0.98) & strong_both)
evaluate(df_val['label'], strat_c, "0.99 OR (0.98 & strong_both)")

# Strategy D: Margin rule
# Calculate rank of candidates per S1 entity
df_val['rank'] = df_val.groupby('s1_entity_id')['pred_prob'].rank(method='first', ascending=False)
# Get top 2 probabilities
top2 = df_val[df_val['rank'] <= 2].pivot(index='s1_entity_id', columns='rank', values='pred_prob')
top2.columns = ['prob_1', 'prob_2']
top2['prob_2'] = top2['prob_2'].fillna(0)
top2['margin'] = top2['prob_1'] - top2['prob_2']
df_val = df_val.merge(top2[['margin']], on='s1_entity_id', how='left')

# Only accept 0.98 if margin is high
strat_d = (df_val['pred_prob'] >= 0.99) | ((df_val['pred_prob'] >= 0.98) & (df_val['margin'] >= 0.1))
evaluate(df_val['label'], strat_d, "0.99 OR (0.98 & margin>=0.1)")

# Strategy E: Source-specific thresholds
is_s2 = df_val['cand_entity_id'].str.startswith('S2')
is_s3 = df_val['cand_entity_id'].str.startswith('S3')
strat_e = (is_s2 & (df_val['pred_prob'] >= 0.985)) | (is_s3 & (df_val['pred_prob'] >= 0.98))
evaluate(df_val['label'], strat_e, "S2>=0.985, S3>=0.98")

# Let's write a grid search for rule combinations to maximize F0.5
print("\n--- 5. Grid Search for Optimal Rule ---")
best_f05 = 0
best_rule = ""
for t_base in [0.98, 0.985, 0.99]:
    for t_relax in [0.95, 0.96, 0.97, 0.98]:
        if t_relax >= t_base: continue
        
        # Rule 1: t_base OR (t_relax AND exact_name)
        pred = (df_val['pred_prob'] >= t_base) | ((df_val['pred_prob'] >= t_relax) & (df_val['name_exact_match'] == 1))
        f05, p, r, fp, fn = evaluate(df_val['label'], pred, f"{t_base} | {t_relax}+exact_name")
        if f05 > best_f05: best_f05 = f05; best_rule = f"{t_base} | {t_relax}+exact_name"

        # Rule 2: t_base OR (t_relax AND strong_both)
        pred = (df_val['pred_prob'] >= t_base) | ((df_val['pred_prob'] >= t_relax) & strong_both)
        f05, p, r, fp, fn = evaluate(df_val['label'], pred, f"{t_base} | {t_relax}+strong_both")
        if f05 > best_f05: best_f05 = f05; best_rule = f"{t_base} | {t_relax}+strong_both"

print(f"\nBest Rule found: {best_rule} with F0.5 = {best_f05:.4f}")
print("\n--- 6. Rank-Based Fallback Strategy ---")
# If we use 0.99 but fallback to rank 1 if > 0.90
for t_base in [0.98, 0.985, 0.99]:
    for t_fallback in [0.90, 0.92, 0.95, 0.96]:
        # base predictions
        pred_base = df_val['pred_prob'] >= t_base
        # fallback for rank 1
        pred_fallback = (df_val['rank'] == 1) & (df_val['pred_prob'] >= t_fallback)
        pred = pred_base | pred_fallback
        f05, p, r, fp, fn = evaluate(df_val['label'], pred, f"{t_base} | fallback R1 > {t_fallback}")

print("\n--- 7. Optimal Base Threshold Search ---")
for t in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98]:
    pred = (df_val['pred_prob'] >= t)
    evaluate(df_val['label'], pred, f"Base {t}")

print("\n--- 8. Multi-condition Decision Rule Search ---")
best_f05 = 0
best_rule = ""
for t_base in [0.94, 0.95, 0.96, 0.97, 0.98]:
    for t_relax in [0.90, 0.92, 0.94, 0.95]:
        if t_relax >= t_base: continue
        
        # Rule 1: t_base OR (t_relax AND name_exact_match)
        pred = (df_val['pred_prob'] >= t_base) | ((df_val['pred_prob'] >= t_relax) & (df_val['name_exact_match'] == 1))
        f05, p, r, fp, fn = evaluate(df_val['label'], pred, f"{t_base} | {t_relax}+exact_name")
        if f05 > best_f05: best_f05 = f05; best_rule = f"{t_base} | {t_relax}+exact_name"

        # Rule 2: t_base OR (t_relax AND strong_both)
        pred = (df_val['pred_prob'] >= t_base) | ((df_val['pred_prob'] >= t_relax) & strong_both)
        f05, p, r, fp, fn = evaluate(df_val['label'], pred, f"{t_base} | {t_relax}+strong_both")
        if f05 > best_f05: best_f05 = f05; best_rule = f"{t_base} | {t_relax}+strong_both"

print(f"\nBest Rule found: {best_rule} with F0.5 = {best_f05:.4f}")
