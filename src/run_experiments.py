import pandas as pd
import numpy as np
import time
import os
import tracemalloc
from collections import defaultdict

# Setup paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, "dataset", "train")
s1_path = os.path.join(train_dir, "train_source1.tsv")
s2_path = os.path.join(train_dir, "train_source2.tsv")
s3_path = os.path.join(train_dir, "train_source3.tsv")
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

def run_experiments():
    print("="*80)
    print("EXPERIMENT 2: FRANCE ANALYSIS")
    print("="*80)
    print("Checking if France exists in S2, S3 or Ground Truth...")
    
    s2_df = pd.read_csv(s2_path, sep='\t', dtype=str, usecols=['entity_id', 'country'])
    s3_df = pd.read_csv(s3_path, sep='\t', dtype=str, usecols=['entity_id', 'country'])
    
    print('Train S2 top countries:')
    print(s2_df['country'].value_counts(dropna=False).head(3))
    print('\nTrain S3 top countries:')
    print(s3_df['country'].value_counts(dropna=False).head(3))
    
    s2_france = set(s2_df[s2_df['country'] == 'France']['entity_id'])
    s3_france = set(s3_df[s3_df['country'] == 'France']['entity_id'])
    france_ids = s2_france | s3_france
    
    print(f"\nFound {len(s2_france)} S2 entities and {len(s3_france)} S3 entities from France.")
    
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
    gt_france_count = 0
    for row in gt_df['matched_entity_ids'].dropna():
        ids = set(row.split(','))
        if len(ids.intersection(france_ids)) > 0:
            gt_france_count += 1
            
    print(f"Number of ground-truth S1 entities matching at least one French S2/S3 entity: {gt_france_count}")
    print("\nSince France does not exist in S1, and essentially 0 in S2/S3 GT, we cannot evaluate France directly on training.")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 & 3 & 4: BLOCKING RECALL & FEATURES & THRESHOLDING")
    print("="*80)
    
    from blocking_v2_2 import BlockingStrategiesV2_2
    from preprocessing import preprocess_chunk
    
    from evaluate_blocking_v2_1 import load_benchmark_data
    
    print("Loading datasets for recall evaluation...")
    s1_eval, s2_eval, s3_eval, gt_eval = load_benchmark_data(base_dir, s1_rows=50000, s2s3_rows=500000)
    
    print(f"Missing address rates in subset - S1: {s1_eval['business_address'].isna().mean():.4f}, S2: {s2_eval['business_address'].isna().mean():.4f}, S3: {s3_eval['business_address'].isna().mean():.4f}")
    
    # Preprocessing already done by load_benchmark_data
    
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    print("Building indexes...")
    blocker.fit_frequencies(s2_eval)
    blocker.fit_frequencies(s3_eval)
    blocker.build_indexes(s2_eval)
    blocker.build_indexes(s3_eval)
    blocker.finalize_indexes()
    
    for k in [50, 100, 150]:
        print(f"\n--- Evaluating K={k} ---")
        t0 = time.time()
        chunk_rows = s1_eval.to_dict('records')
        batch_cands = blocker.retrieve_and_score_batch(chunk_rows, k=k)
        candidates = {}
        for r_row, cands in zip(chunk_rows, batch_cands):
            candidates[r_row['entity_id']] = [c[0] for c in cands]
        
        overall_hits, s2_hits, s3_hits = 0, 0, 0
        overall_possible, s2_possible, s3_possible = 0, 0, 0
        both_hits, both_possible = 0, 0
        missing_addr_hits, missing_addr_possible = 0, 0
        
        for idx, row in s1_eval.iterrows():
            s1_id = row['entity_id']
            cand_set = set(candidates.get(s1_id, []))
            
            gt_row = gt_eval[gt_eval['source1_entity_id'] == s1_id]
            if gt_row.empty or pd.isna(gt_row.iloc[0]['matched_entity_ids']):
                continue
                
            true_matches = set(gt_row.iloc[0]['matched_entity_ids'].split(','))
            if not true_matches: continue
            
            s2_true = {m for m in true_matches if m.startswith('S2')}
            s3_true = {m for m in true_matches if m.startswith('S3')}
            
            overall_possible += len(true_matches)
            overall_hits += len(cand_set.intersection(true_matches))
            
            if s2_true:
                s2_possible += len(s2_true)
                s2_hits += len(cand_set.intersection(s2_true))
            if s3_true:
                s3_possible += len(s3_true)
                s3_hits += len(cand_set.intersection(s3_true))
                
            if s2_true and s3_true:
                both_possible += 1
                if cand_set.intersection(s2_true) and cand_set.intersection(s3_true):
                    both_hits += 1
                    
            if pd.isna(row['business_address']) or str(row['business_address']) == '' or str(row['business_address']).lower() == 'nan':
                missing_addr_possible += len(true_matches)
                missing_addr_hits += len(cand_set.intersection(true_matches))
                
        print(f"Overall Recall: {overall_hits/max(1, overall_possible):.4f} ({overall_hits}/{overall_possible})")
        print(f"S2 Recall: {s2_hits/max(1, s2_possible):.4f}")
        print(f"S3 Recall: {s3_hits/max(1, s3_possible):.4f}")
        print(f"Both-Source Recall (entity level): {both_hits/max(1, both_possible):.4f}")
        print(f"Missing-Address Recall: {missing_addr_hits/max(1, missing_addr_possible):.4f}")
        print(f"Time taken: {time.time()-t0:.2f}s")
        
        if k == 50:
            print("\nEvaluating Adaptive Thresholding on K=50 candidates...")
            try:
                from matching_features import PairFeatureGenerator
                import lightgbm as lgb
                generator = PairFeatureGenerator(idf_dict=blocker.idf)
                model = lgb.Booster(model_file='../models/lgbm_model.txt')
                feature_cols = model.feature_name()
                
                rows = []
                for s1_id, cands in candidates.items():
                    for cand_id in cands:
                        rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})
                cand_df = pd.DataFrame(rows)
                
                print(f"Generated {len(cand_df)} candidate pairs. Extracting features...")
                s1_meta = s1_eval.set_index('entity_id')
                s2_meta = s2_eval.set_index('entity_id')
                s3_meta = s3_eval.set_index('entity_id')
                
                s1_meta_renamed = s1_meta.add_suffix('_s1')
                s2s3_meta = pd.concat([s2_meta, s3_meta])
                s2s3_meta_renamed = s2s3_meta.add_suffix('_s2s3')
                
                merged = pd.merge(cand_df, s1_meta_renamed, left_on='source1_entity_id', right_index=True)
                merged = pd.merge(merged, s2s3_meta_renamed, left_on='candidate_entity_id', right_index=True)
                
                s1_cols = [c for c in merged.columns if c.endswith('_s1')]
                s2s3_cols = [c for c in merged.columns if c.endswith('_s2s3')]
                
                s1_df_batch = merged[['source1_entity_id'] + s1_cols].rename(columns={'source1_entity_id': 'entity_id'})
                s1_df_batch.columns = [c.replace('_s1', '') for c in s1_df_batch.columns]
                
                s2_df_batch = merged[['candidate_entity_id'] + s2s3_cols].rename(columns={'candidate_entity_id': 'entity_id'})
                s2_df_batch.columns = [c.replace('_s2s3', '') for c in s2_df_batch.columns]
                
                b_scores = np.zeros(len(merged))
                b_ranks = np.zeros(len(merged))
                
                features_df = generator.generate_features_batch(s1_df_batch, s2_df_batch, b_scores, b_ranks)
                for col in features_df.columns:
                    merged[col] = features_df[col].values
                X = features_df[feature_cols]
                preds = model.predict(X)
                merged['pred'] = preds
                
                merged['is_true'] = 0
                for idx, row in gt_eval.iterrows():
                    s1 = row['source1_entity_id']
                    if pd.notna(row['matched_entity_ids']):
                        ms = set(row['matched_entity_ids'].split(','))
                        mask = (merged['source1_entity_id'] == s1) & (merged['candidate_entity_id'].isin(ms))
                        merged.loc[mask, 'is_true'] = 1
                
                thresholds = [0.90, 0.95, 0.97, 0.98]
                print("\nThresholding Experiments:")
                for t in thresholds:
                    accepted = merged[merged['pred'] >= t]
                    tp = accepted['is_true'].sum()
                    fp = len(accepted) - tp
                    precision = tp / max(1, (tp + fp))
                    recall = tp / max(1, overall_possible)
                    f05 = 1.25 * (precision * recall) / max(1e-9, 0.25 * precision + recall)
                    empty_s1 = s1_eval['entity_id'].nunique() - accepted['source1_entity_id'].nunique()
                    print(f"Threshold {t}: P={precision:.4f}, R={recall:.4f}, F0.5={f05:.4f}, Empty S1={empty_s1}")
                    
                print("\nAdaptive Rule Experiment:")
                accepted_adapt = merged[merged['pred'] >= 0.98].copy()
                accepted_s1s = set(accepted_adapt['source1_entity_id'])
                
                zero_match_s1s = set(s1_eval['entity_id']) - accepted_s1s
                fallback_candidates = merged[merged['source1_entity_id'].isin(zero_match_s1s)]
                
                if not fallback_candidates.empty:
                    fallback_max_idx = fallback_candidates.groupby('source1_entity_id')['pred'].idxmax()
                    fallback_max = fallback_candidates.loc[fallback_max_idx]
                    
                    condition = (
                        (fallback_max['name_edit_sim'] > 0.9) | 
                        ((fallback_max['name_token_jaccard'] > 0.7) & (fallback_max['addr_token_jaccard'] > 0.7)) |
                        ((fallback_max['postal_exact_match'] == 1) & (fallback_max['name_token_jaccard'] > 0.6))
                    )
                    fallback_accepted = fallback_max[condition]
                    final_adapt = pd.concat([accepted_adapt, fallback_accepted])
                else:
                    final_adapt = accepted_adapt
                    
                tp = final_adapt['is_true'].sum()
                fp = len(final_adapt) - tp
                precision = tp / max(1, (tp + fp))
                recall = tp / max(1, overall_possible)
                f05 = 1.25 * (precision * recall) / max(1e-9, 0.25 * precision + recall)
                empty_s1 = s1_eval['entity_id'].nunique() - final_adapt['source1_entity_id'].nunique()
                print(f"Adaptive Rule: P={precision:.4f}, R={recall:.4f}, F0.5={f05:.4f}, Empty S1={empty_s1}")
                
            except Exception as e:
                print(f"Error in evaluation: {e}")

if __name__ == '__main__':
    run_experiments()
