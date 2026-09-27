import os
import sys
import time
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_chunk as preprocess_pipeline
from blocking_v2_2 import BlockingStrategiesV2_2
from matching_features import PairFeatureGenerator

def load_validation_data(base_dir):
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    val_preds = pd.read_parquet(os.path.join(base_dir, "models", "val_predictions.parquet"))
    
    val_s1_ids = list(val_preds['s1_entity_id'].unique())
    # ONLY TAKE 2000 for speed
    val_s1_ids = set(val_s1_ids[:2000])
    
    print(f"Loading {len(val_s1_ids)} S1 entities from validation set...")
    
    s1_df = pd.read_csv(s1_path, sep='\t')
    s1_df = s1_df[s1_df['entity_id'].isin(val_s1_ids)]
    s1_df['address_missing'] = s1_df['business_address'].isna() | (s1_df['business_address'] == "")
    s1_df = preprocess_pipeline(s1_df)
    
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_df = gt_df[gt_df['source1_entity_id'].isin(val_s1_ids)]
    
    true_matches = set()
    for matches in gt_df['matched_entity_ids']:
        if pd.isna(matches): continue
        true_matches.update(matches.split(','))
    
    s2s3_rows = 500000
    
    def load_source(path, required_ids):
        df_list = []
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=100000)
        remaining = s2s3_rows
        for chunk in chunk_iter:
            chunk = preprocess_pipeline(chunk)
            if remaining > 0:
                keep = chunk.head(remaining)
                df_list.append(keep)
                remaining -= len(keep)
            req_chunk = chunk[chunk['entity_id'].isin(required_ids)]
            if not req_chunk.empty:
                df_list.append(req_chunk)
        return pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['entity_id'])

    print("Loading Source 2 and 3...")
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s2_true = {x for x in true_matches if x.startswith('S2-')}
    s2_df = load_source(s2_path, s2_true)
    
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    s3_true = {x for x in true_matches if x.startswith('S3-')}
    s3_df = load_source(s3_path, s3_true)
    
    return s1_df, s2_df, s3_df, gt_df

def calc_f05_vectorized(df, pred_col):
    agg = df.groupby('s1_entity_id').agg(
        TP=('label', lambda y: ((y == 1) & (df.loc[y.index, pred_col] == 1)).sum()),
        FP=('label', lambda y: ((y == 0) & (df.loc[y.index, pred_col] == 1)).sum()),
        FN=('label', lambda y: ((y == 1) & (df.loc[y.index, pred_col] == 0)).sum()),
        True_Count=('label', 'sum'),
        Pred_Count=(pred_col, 'sum')
    )
    P = np.where(agg['TP'] + agg['FP'] > 0, agg['TP'] / (agg['TP'] + agg['FP']), 0)
    R = np.where(agg['TP'] + agg['FN'] > 0, agg['TP'] / (agg['TP'] + agg['FN']), 0)
    f05 = np.zeros(len(agg))
    both_empty = (agg['True_Count'] == 0) & (agg['Pred_Count'] == 0)
    f05[both_empty] = 1.0
    mask = (P + R > 0) & (~both_empty)
    f05[mask] = (1.25 * P[mask] * R[mask]) / (0.25 * P[mask] + R[mask])
    return f05.mean()

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_df, s2_df, s3_df, gt_df = load_validation_data(base_dir)
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
        gt_dict[row['source1_entity_id']] = set(matches)
        
    print("Building blocking index V2.2...")
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    blocker.finalize_indexes()
    
    print("Loading LightGBM model...")
    model = lgb.Booster(model_file=os.path.join(base_dir, 'models', 'lgbm_model.txt'))
    feature_generator = PairFeatureGenerator(idf_dict=blocker.idf)
    
    cand_dfs = {s2_df.iloc[i]['entity_id']: s2_df.iloc[i] for i in range(len(s2_df))}
    for i in range(len(s3_df)): cand_dfs[s3_df.iloc[i]['entity_id']] = s3_df.iloc[i]
    
    k_values = [150, 250, 300, 500]
    
    print("Evaluating end-to-end strategies...")
    rows = list(s1_df.itertuples(index=False))
    k_max = max(k_values)
    
    all_s1, all_s23, all_labels, all_s1_ids, all_cand_ids, all_blocking_scores, all_blocking_ranks = [], [], [], [], [], [], []
    
    for row in rows:
        scored_cands = blocker.retrieve_and_score(row, k=k_max)
        true_matches = gt_dict.get(row.entity_id, set())
        for r_idx, (cand_id, b_score) in enumerate(scored_cands):
            cand_row = cand_dfs.get(cand_id)
            if cand_row is not None:
                all_s1.append(row)
                all_s23.append(cand_row)
                all_labels.append(1 if cand_id in true_matches else 0)
                all_s1_ids.append(row.entity_id)
                all_cand_ids.append(cand_id)
                all_blocking_scores.append(b_score)
                all_blocking_ranks.append(r_idx + 1)

    print("Generating features...")
    df_s1 = pd.DataFrame(all_s1)
    df_s23 = pd.DataFrame(all_s23)
    X = feature_generator.generate_features_batch(df_s1, df_s23, np.array(all_blocking_scores), np.array(all_blocking_ranks))
    
    print("Predicting...")
    preds = model.predict(X[model.feature_name()])
    
    results_df = pd.DataFrame({
        's1_entity_id': all_s1_ids,
        'cand_entity_id': all_cand_ids,
        'label': all_labels,
        'pred_prob': preds
    })
    
    results_df['rank'] = results_df.groupby('s1_entity_id').cumcount() + 1
    
    print("\n--- RESULTS ---")
    for k in k_values:
        df_k = results_df[results_df['rank'] <= k].copy()
        cand_recall = df_k['label'].sum() / sum(len(v) for v in gt_dict.values())
        
        df_k.loc[:, 'pred'] = (df_k['pred_prob'] >= 0.98).astype(int)
        
        missing_s1 = set(s1_df['entity_id']) - set(df_k['s1_entity_id'])
        missing_df = pd.DataFrame({
            's1_entity_id': list(missing_s1),
            'cand_entity_id': 'NONE',
            'label': 0,
            'pred': 0
        })
        for idx, row in missing_df.iterrows():
            if len(gt_dict.get(row['s1_entity_id'], set())) > 0:
                missing_df.at[idx, 'label'] = 1
                
        eval_df = pd.concat([df_k, missing_df])
        
        f05 = calc_f05_vectorized(eval_df, 'pred')
        p = precision_score(eval_df['label'], eval_df['pred'], zero_division=0)
        r = recall_score(eval_df['label'], eval_df['pred'], zero_division=0)
        
        print(f"K={k:<3} | Cand Recall: {cand_recall:.4f} | Cand Count: {len(df_k):,} | P: {p:.4f} | R: {r:.4f} | Entity F0.5: {f05:.4f}")

if __name__ == "__main__":
    evaluate()
