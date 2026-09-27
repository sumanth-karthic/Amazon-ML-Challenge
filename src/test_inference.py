import os
import time
import psutil
import pandas as pd
import numpy as np
import lightgbm as lgb
from preprocessing import preprocess_chunk
from blocking_v2_2 import BlockingStrategiesV2_2
from matching_features import PairFeatureGenerator

def log_ram(msg):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"{msg}. RAM: {mem:.1f} MB")
    return mem

def run_test_inference():
    t0 = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    # 1. Load Datasets
    print("Loading Test Source 2...")
    cols_to_use = ['entity_id', 'business_name', 'business_address', 'country']
    s2_df = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', dtype=str, usecols=cols_to_use).fillna('')
    s2_df = preprocess_chunk(s2_df)
    
    print("Loading Test Source 3...")
    s3_df = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', dtype=str, usecols=cols_to_use).fillna('')
    s3_df = preprocess_chunk(s3_df)
    
    peak_ram = log_ram("S2/S3 Data loaded")
    
    # 2. Build Sparse Blocker
    print("Building V2.2 Sparse Index on Test S2/S3...")
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    blocker.finalize_indexes()
    
    peak_ram = max(peak_ram, log_ram("Blocker matrix built"))
    
    # Store S2/S3 metadata in DataFrame for fast lookup (Memory efficient)
    print("Building S2/S3 Metadata DataFrame for fast lookup...")
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    s2s3_df.set_index('entity_id', inplace=True)
        
    import gc
    del s2_df
    del s3_df
    gc.collect()
        
    peak_ram = max(peak_ram, log_ram("DataFrame built and dfs deleted"))
    
    print("Loading Test Source 1...")
    s1_df = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', dtype=str, usecols=cols_to_use).fillna('')
    s1_df = preprocess_chunk(s1_df)
    s1_cols = list(s1_df.columns)
    peak_ram = max(peak_ram, log_ram("S1 Data loaded"))
    
    # Load Model to get exact feature order
    model_path = os.path.join(base_dir, "models", "lgbm_model.txt")
    model = lgb.Booster(model_file=model_path)
    feature_cols = model.feature_name()
    print(f"Model expects exactly {len(feature_cols)} features: {feature_cols}")
    
    # 3. Stream Candidate Pairs and Run Inference
    print("\nStarting Streaming Feature Generation and Inference...")
    cand_pairs_path = os.path.join(base_dir, "candidate_pairs.tsv")
    
    generator = PairFeatureGenerator(idf_dict=blocker.idf)
    chunk_size = 5000
    
    total_pairs = 0
    t_feat_total = 0
    t_inf_total = 0
    t_block_total = 0
    
    # Final results dictionary
    predicted_matches = {}
    
    with open(cand_pairs_path, 'w', encoding='utf-8') as cp_out:
        cp_out.write("s1_entity_id\tcand_entity_id\n")
        
        s1_rows = list(s1_df.itertuples(index=False))
        
        for i in range(0, len(s1_rows), chunk_size):
            chunk_rows = s1_rows[i : i + chunk_size]
            
            t_b_start = time.time()
            batch_cands = blocker.retrieve_and_score_batch(chunk_rows, k=50)
            t_block_total += (time.time() - t_b_start)
            
            t_f_start = time.time()
            
            s1_replicas = []
            flat_cand_ids = []
            blocking_scores = []
            blocking_ranks = []
            s1_ids = []
            
            for s1_row, cands in zip(chunk_rows, batch_cands):
                if not cands: continue
                for rank, (cand_id, score) in enumerate(cands, 1):
                    s1_replicas.append(s1_row)
                    s1_ids.append(s1_row[0])  # entity_id is 0th column
                    flat_cand_ids.append(cand_id)
                    blocking_scores.append(score)
                    blocking_ranks.append(rank)
            
            if not flat_cand_ids:
                continue
                
            s1_chunk_df = pd.DataFrame(s1_replicas, columns=s1_cols)
            # single fast metadata fetch for all candidates in the batch
            s2_chunk_df = s2s3_df.loc[flat_cand_ids].reset_index(drop=True)
            
            # vectorized feature generation
            chunk_df = generator.generate_features_batch(s1_chunk_df, s2_chunk_df, blocking_scores, blocking_ranks)
            
            # Add entity ids
            chunk_df.insert(0, 'cand_entity_id', flat_cand_ids)
            chunk_df.insert(0, 's1_entity_id', s1_ids)
            
            t_feat_total += (time.time() - t_f_start)
            
            # Ensure columns order
            X_chunk = chunk_df[feature_cols]
            
            t_i_start = time.time()
            preds = model.predict(X_chunk)
            t_inf_total += (time.time() - t_i_start)
            
            chunk_df['pred_prob'] = preds
            matches_df = chunk_df[chunk_df['pred_prob'] >= 0.90]
            
            # Accumulate matches
            for r in matches_df.itertuples(index=False):
                if r.s1_entity_id not in predicted_matches:
                    predicted_matches[r.s1_entity_id] = []
                predicted_matches[r.s1_entity_id].append(r.cand_entity_id)
            
            # Write to candidate pairs tsv
            for r in chunk_df.itertuples(index=False):
                cp_out.write(f"{r.s1_entity_id}\t{r.cand_entity_id}\n")
            cp_out.flush()
                
            total_pairs += len(chunk_df)
                
            peak_ram = max(peak_ram, log_ram(f"Processed {min(i+chunk_size, len(s1_rows))}/{len(s1_rows)} S1 entities. Pairs: {total_pairs}"))

    # 4. Finalize Matching Results TSV
    print("\nBuilding matching_results.tsv...")
    final_output = []
    
    total_predicted_matches = 0
    s1_with_matches = 0
    s1_singletons = 0
    s2_matches = 0
    s3_matches = 0
    
    for s1_id in s1_df['entity_id']:
        cand_list = predicted_matches.get(s1_id, [])
        cand_list.sort() # deterministic ordering
        
        if cand_list:
            s1_with_matches += 1
            total_predicted_matches += len(cand_list)
            for c in cand_list:
                if c.startswith('S2-'): s2_matches += 1
                elif c.startswith('S3-'): s3_matches += 1
        else:
            s1_singletons += 1
            
        cand_str = " ".join(cand_list)
        final_output.append({'s1_entity_id': s1_id, 'matched_entity_ids': cand_str})
        
    results_df = pd.DataFrame(final_output)
    
    results_path = os.path.join(base_dir, "matching_results.tsv")
    results_df.to_csv(results_path, sep='\t', index=False)
    
    print(f"Saved inference results to: {results_path}")
    print(f"Saved actual candidate pairs to: {cand_pairs_path}")
    
    file_size_mb_cp = os.path.getsize(cand_pairs_path) / (1024 * 1024)
    file_size_mb_mr = os.path.getsize(results_path) / (1024 * 1024)
    
    # 5. Report Statistics
    total_time = time.time() - t0
    print("\n================================================================================")
    print("FINAL TEST INFERENCE REPORT")
    print("================================================================================")
    print(f"Total Test S1 Entities:           {len(s1_df):,}")
    print(f"Total Candidate Pairs:            {total_pairs:,}")
    print(f"Average Candidates per S1:        {total_pairs/len(s1_df):.2f}")
    print(f"Number of Predicted Matches:      {total_predicted_matches:,}")
    print(f"S1 Entities w/ >=1 Match:         {s1_with_matches:,}")
    print(f"S1 Entities w/ 0 Matches (Sing):  {s1_singletons:,}")
    print(f"S2 Predicted Matches:             {s2_matches:,}")
    print(f"S3 Predicted Matches:             {s3_matches:,}")
    print(f"Peak RAM:                         {peak_ram:.1f} MB")
    print(f"Total Runtime:                    {total_time:.2f} seconds")
    print(f"Blocker Retrieval Time:           {t_block_total:.2f} seconds")
    print(f"Feature Generation Time:          {t_feat_total:.2f} seconds")
    print(f"Inference Time:                   {t_inf_total:.2f} seconds")
    print(f"File Size (candidate_pairs.tsv):  {file_size_mb_cp:.2f} MB")
    print(f"File Size (matching_results.tsv): {file_size_mb_mr:.2f} MB")
    print("================================================================================")

if __name__ == '__main__':
    run_test_inference()
