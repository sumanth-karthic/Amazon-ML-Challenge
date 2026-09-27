import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

print("Loading train_features.parquet...")
df = pd.read_parquet('dataset/train_features.parquet')

print("Loading val_predictions.parquet...")
val_preds = pd.read_parquet('models/val_predictions.parquet')

df_val = df.merge(val_preds[['s1_entity_id', 'cand_entity_id', 'pred_prob']], on=['s1_entity_id', 'cand_entity_id'], how='inner')
print(f"Validation set size: {len(df_val):,} pairs")

# Load country data for S1
print("Loading train_source1.tsv for country breakdown...")
s1_data = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id', 'country'])
s1_data = s1_data.rename(columns={'entity_id': 's1_entity_id', 'country': 's1_country'})
df_val = df_val.merge(s1_data, on='s1_entity_id', how='left')

# Base 0.98 predictions
pred_98 = (df_val['pred_prob'] >= 0.98).astype(int)
# Exact Name rule predictions
pred_exact = (df_val['pred_prob'] >= 0.98) | ((df_val['pred_prob'] >= 0.96) & (df_val['name_exact_match'] == 1))
pred_exact = pred_exact.astype(int)

df_val['pred_98'] = pred_98
df_val['pred_exact'] = pred_exact

print("\n=== 1. Additional Predictions ===")
added = (pred_exact == 1) & (pred_98 == 0)
num_added = added.sum()
print(f"Total additional predictions added by the exact-name rule (0.96 vs 0.98): {num_added}")

print("\n=== 2. TP vs FP of Additional Predictions ===")
added_tp = ((added) & (df_val['label'] == 1)).sum()
added_fp = ((added) & (df_val['label'] == 0)).sum()
print(f"Of the {num_added} added: {added_tp} are True Positives, {added_fp} are False Positives")

print("\n=== 3, 6, 7. Entity-Level Match Changes ===")
# Number of matches predicted per entity under both rules
matches_98 = df_val.groupby('s1_entity_id')['pred_98'].sum()
matches_exact = df_val.groupby('s1_entity_id')['pred_exact'].sum()
true_matches = df_val.groupby('s1_entity_id')['label'].sum()

# S1 entities getting their FIRST/ONLY match because of the rule
# This means matches_98 == 0 AND matches_exact > 0
first_match_mask = (matches_98 == 0) & (matches_exact > 0)
num_first_match = first_match_mask.sum()
print(f"Entities going from 0 matches -> at least 1 match: {num_first_match}")
print(f"Percentage of total S1 entities receiving first/only match: {num_first_match / len(matches_98) * 100:.4f}%")

# Entities going from correct -> incorrect (F0.5 drops)
# F0.5 per entity
def f05_entity(tp, fp, fn):
    # if true is empty and pred is empty => 1.0
    # if true is empty and pred not empty => 0.0
    # if true not empty and pred is empty => 0.0
    pass

# Let's do it vectorized
def calc_f05_vectorized(df, pred_col):
    agg = df.groupby('s1_entity_id').agg(
        TP=('label', lambda y: ((y == 1) & (df.loc[y.index, pred_col] == 1)).sum()),
        FP=('label', lambda y: ((y == 0) & (df.loc[y.index, pred_col] == 1)).sum()),
        FN=('label', lambda y: ((y == 1) & (df.loc[y.index, pred_col] == 0)).sum()),
        True_Count=('label', 'sum'),
        Pred_Count=(pred_col, 'sum')
    )
    
    # Calculate F0.5 per entity
    P = np.where(agg['TP'] + agg['FP'] > 0, agg['TP'] / (agg['TP'] + agg['FP']), 0)
    R = np.where(agg['TP'] + agg['FN'] > 0, agg['TP'] / (agg['TP'] + agg['FN']), 0)
    
    # Handle singletons
    f05 = np.zeros(len(agg))
    
    # Both empty -> 1.0
    both_empty = (agg['True_Count'] == 0) & (agg['Pred_Count'] == 0)
    f05[both_empty] = 1.0
    
    # Calculate for non-empty
    mask = (P + R > 0) & (~both_empty)
    f05[mask] = (1.25 * P[mask] * R[mask]) / (0.25 * P[mask] + R[mask])
    
    return f05, agg

f05_98, agg_98 = calc_f05_vectorized(df_val, 'pred_98')
f05_exact, agg_exact = calc_f05_vectorized(df_val, 'pred_exact')

drops = (f05_exact < f05_98).sum()
improves = (f05_exact > f05_98).sum()
print(f"Entities that went from correct -> incorrect (F0.5 decreased): {drops}")
print(f"Entities that went from incorrect -> correct (F0.5 increased): {improves}")
print(f"Macro F0.5 (Base 0.98): {f05_98.mean():.6f}")
print(f"Macro F0.5 (Exact-Rule): {f05_exact.mean():.6f}")


print("\n=== 4. Breakdown by Source 2 vs Source 3 ===")
df_added = df_val[added]
s2_added = df_added['cand_entity_id'].str.startswith('S2').sum()
s3_added = df_added['cand_entity_id'].str.startswith('S3').sum()
print(f"Added S2 predictions: {s2_added}")
print(f"Added S3 predictions: {s3_added}")


print("\n=== 5. Breakdown by US vs India ===")
us_added = df_added[df_added['s1_country'] == 'US'].shape[0]
in_added = df_added[df_added['s1_country'] == 'IN'].shape[0]
print(f"Added US predictions: {us_added}")
print(f"Added India predictions: {in_added}")


print("\n=== 8 & 9. Threshold Sweep ===")
for t in [0.94, 0.95, 0.96, 0.97, 0.98]:
    pred_col = f'pred_exact_{t}'
    df_val[pred_col] = (df_val['pred_prob'] >= 0.98) | ((df_val['pred_prob'] >= t) & (df_val['name_exact_match'] == 1))
    f05_t, _ = calc_f05_vectorized(df_val, pred_col)
    
    p = precision_score(df_val['label'], df_val[pred_col], zero_division=0)
    r = recall_score(df_val['label'], df_val[pred_col], zero_division=0)
    print(f"Exact Rule @ {t:.2f} | Entity-Macro F0.5: {f05_t.mean():.6f} | Micro P: {p:.4f} | Micro R: {r:.4f}")

print("\n=== 10. Estimate of Additional TEST Predictions ===")
# We have matching_098.tsv with exactly 1,460,941 positive predictions (or some number).
# Let's find the ratio of additional predictions to base 0.98 predictions in Validation
val_base_preds = pred_98.sum()
val_added = num_added
ratio = val_added / val_base_preds
print(f"Validation Base Predictions: {val_base_preds}")
print(f"Validation Added Predictions: {val_added} ({ratio*100:.2f}%)")

# Now check test predictions
try:
    with open('output/threshold_tests/matching_098.tsv', 'r') as f:
        # count lines that have a comma or are just one prediction?
        # Since it's grouped, we can just sum the commas + 1 for each line that has matches
        test_preds = 0
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and parts[1]:
                test_preds += parts[1].count(',') + 1
    est_added = test_preds * ratio
    print(f"Test Base Predictions (0.98): {test_preds}")
    print(f"Estimated Added Test Predictions: ~{int(est_added)}")
except Exception as e:
    print("Could not calculate test estimate: ", e)
