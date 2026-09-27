import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score
import os

print("--- 1-3. Validation Set Metrics ---")
df_val = pd.read_csv('../dataset/train/train_features_val.csv')
model = lgb.Booster(model_file='../models/lgbm_model.txt')
y_true = df_val['label']
X_val = df_val[model.feature_name()]
preds = model.predict(X_val)

thresholds = [0.98, 0.99, 0.995]
for th in thresholds:
    y_pred = (preds >= th).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    print(f"Threshold {th}: F0.5={f05:.4f}, Precision={p:.4f}, Recall={r:.4f}, FP={fp}, FN={fn}")

print("\n--- 4. Pseudo-Unseen US/India Calibration ---")
# Train model on US only
df_train = pd.read_csv('../dataset/train/train_features_train.csv')
s1_train = pd.read_csv('../dataset/train/train_source1.tsv', sep='\t', dtype=str).fillna('')
us_ids = s1_train[s1_train['country'] == 'us']['entity_id'].values
in_ids = s1_train[s1_train['country'] == 'in']['entity_id'].values

us_mask = df_train['s1_entity_id'].isin(us_ids)
in_mask = df_train['s1_entity_id'].isin(in_ids)

model_us = lgb.train(
    {'objective': 'binary', 'metric': 'binary_error'},
    lgb.Dataset(df_train[us_mask][model.feature_name()], label=df_train[us_mask]['label'])
)

preds_in = model_us.predict(df_train[in_mask][model.feature_name()])
y_true_in = df_train[in_mask]['label']
for th in thresholds:
    y_pred = (preds_in >= th).astype(int)
    p = precision_score(y_true_in, y_pred, zero_division=0)
    r = recall_score(y_true_in, y_pred)
    f05 = fbeta_score(y_true_in, y_pred, beta=0.5)
    print(f"US->India (Unseen) @ {th}: F0.5={f05:.4f}, Precision={p:.4f}, Recall={r:.4f}")

print("\n--- 5-7. Test Predictions Analysis ---")
s1_test = pd.read_csv('../dataset/test/test_source1.tsv', sep='\t', dtype=str).fillna('')

res_098 = pd.read_csv('../output/threshold_tests/matching_098.tsv', sep='\t', names=['source1_entity_id', 'matched_entity_ids'])
res_098 = s1_test[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'}).merge(res_098, on='source1_entity_id', how='left').fillna('')

res_099 = pd.read_csv('../output/threshold_tests/matching_results_0.99.tsv', sep='\t', dtype=str).fillna('')
res_0995 = pd.read_csv('../output/threshold_tests/matching_results_0.995.tsv', sep='\t', dtype=str).fillna('')

def analyze_matches(df, th_name):
    # count matches
    df['match_count'] = df['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if str(x).strip() != '' else 0)
    df_merged = df.merge(s1_test[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    country_counts = df_merged[df_merged['match_count'] > 0].groupby('country').size()
    avg_matches = df['match_count'].mean()
    max_matches = df['match_count'].max()
    zero_matches = (df['match_count'] == 0).sum()
    print(f"\n[{th_name}]")
    print(f"Average Matches per S1: {avg_matches:.4f}")
    print(f"Max Matches for single S1: {max_matches}")
    print(f"Zero Match S1s: {zero_matches} ({(zero_matches/len(df))*100:.2f}%)")
    print("Match counts by Country:")
    for c, count in country_counts.items():
        print(f"  {c}: {count}")
    return df['matched_entity_ids'].values

m98 = analyze_matches(res_098, "Threshold 0.98")
m99 = analyze_matches(res_099, "Threshold 0.99")
m995 = analyze_matches(res_0995, "Threshold 0.995")

changed_98_99 = np.sum(m98 != m99)
changed_99_995 = np.sum(m99 != m995)
print(f"\nEntities changed from 0.98 to 0.99: {changed_98_99}")
print(f"Entities changed from 0.99 to 0.995: {changed_99_995}")
