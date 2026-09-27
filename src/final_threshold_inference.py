import pandas as pd
import numpy as np
import time
import os
import gc
import scipy.sparse as sp
import lightgbm as lgb
from collections import defaultdict

from preprocessing import preprocess_chunk
from matching_features import PairFeatureGenerator
from blocking_v2_2 import BlockingStrategiesV2_2

def run_threshold_inference():
    print("Loading datasets...")
    test_dir = '../dataset/test'
    s1_df = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', dtype=str).fillna('')
    s2_df = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', dtype=str).fillna('')
    s3_df = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', dtype=str).fillna('')

    s1_df = preprocess_chunk(s1_df)
    s2_df = preprocess_chunk(s2_df)
    s3_df = preprocess_chunk(s3_df)

    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)

    print("Building fast TF-IDF index for blocking scores...")
    t_start = time.time()
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    blocker.finalize_indexes()
    
    ent2idx = {eid: i for i, eid in enumerate(blocker.entity_ids)}

    s2s3_df.set_index('entity_id', inplace=True)
    
    def score_pairs_fast(s1_rows, cand_ids):
        # Extracts specific candidate columns
        c_indices = [ent2idx[cid] for cid in cand_ids]
        C_batch = blocker.matrix[:, c_indices]
        
        q_data, q_row, q_col = [], [], []
        for i, s1_row in enumerate(s1_rows):
            name_tokens = getattr(s1_row, 'name_norm', '').split()
            addr_tokens = getattr(s1_row, 'address_norm', '').split()
            ngrams = getattr(s1_row, 'ngrams', [])
            postal = getattr(s1_row, 'postal_code', '')
            exact = getattr(s1_row, 'name_norm', '')
            
            features = set()
            if exact: features.add(f"E:{exact}")
            for nt in name_tokens: features.add(f"N:{nt}")
            for at in addr_tokens: features.add(f"A:{at}")
            for ng in ngrams[:blocker.num_ngrams]: features.add(f"G:{ng}")
            if postal:
                for nt in name_tokens[:3]: features.add(f"P:{postal}|{nt}")
            for nt in name_tokens[:2]:
                for at in addr_tokens[:2]: features.add(f"NA:{nt}|{at}")
                    
            for feat in features:
                v_id = blocker.vocab.get(feat)
                if v_id is not None:
                    q_row.append(i)
                    q_col.append(v_id)
                    q_data.append(1.0)
                    
        Q_batch = sp.csr_matrix((q_data, (q_row, q_col)), shape=(len(s1_rows), blocker.vocab_size), dtype=np.float32)
        scores = Q_batch.multiply(C_batch.T).sum(axis=1).A1
        return scores

    print("Loading LightGBM model...")
    model = lgb.Booster(model_file='../models/lgbm_model.txt')
    generator = PairFeatureGenerator(idf_dict=blocker.idf)
    feature_cols = model.feature_name()

    os.makedirs('../output/threshold_tests', exist_ok=True)
    
    chunk_size = 500000
    
    matches_099 = defaultdict(list)
    matches_0995 = defaultdict(list)
    
    total_pairs = 0
    t_inference = time.time()
    
    print("Streaming candidate pairs...")
    s1_cols = list(s1_df.columns)
    s1_df_indexed = s1_df.set_index('entity_id')
    
    # Read chunk by chunk
    reader = pd.read_csv('../candidate_pairs.tsv', sep='\t', chunksize=chunk_size, dtype=str)
    
    current_s1 = None
    current_rank = 0
    
    for chunk in reader:
        s1_ids = chunk['source1_entity_id'].values
        cand_ids = chunk['candidate_entity_ids'].values
        
        # We need to construct s1_rows to pass to score_pairs_fast and generator
        s1_chunk_df = s1_df_indexed.loc[s1_ids].reset_index()
        s2_chunk_df = s2s3_df.loc[cand_ids].reset_index()
        
        s1_rows_tuples = list(s1_chunk_df.itertuples(index=False))
        
        # Calculate blocking score
        scores = score_pairs_fast(s1_rows_tuples, cand_ids)
        
        # Calculate blocking rank (sequential rank per S1)
        # We can just count occurrences
        ranks = []
        for s1_id in s1_ids:
            if s1_id != current_s1:
                current_s1 = s1_id
                current_rank = 1
            else:
                current_rank += 1
            ranks.append(current_rank)
            
        # Generate features
        feat_df = generator.generate_features_batch(s1_chunk_df, s2_chunk_df, scores, ranks)
        X_chunk = feat_df[feature_cols]
        
        preds = model.predict(X_chunk)
        
        # Filter predictions >= 0.99
        mask_099 = preds >= 0.99
        for s1_id, cand_id in zip(s1_ids[mask_099], cand_ids[mask_099]):
            matches_099[s1_id].append(cand_id)
            
        # Filter predictions >= 0.995
        mask_0995 = preds >= 0.995
        for s1_id, cand_id in zip(s1_ids[mask_0995], cand_ids[mask_0995]):
            matches_0995[s1_id].append(cand_id)
            
        total_pairs += len(chunk)
        print(f"Processed {total_pairs} pairs...")
        
    print("Inference completed. Writing final results...")
    
    # We must write EXACTLY 1,732,544 rows, preserving order of S1
    with open('../output/threshold_tests/matching_results_0.99.tsv', 'w', encoding='utf-8') as f99, \
         open('../output/threshold_tests/matching_results_0.995.tsv', 'w', encoding='utf-8') as f995:
         
        f99.write("source1_entity_id\tmatched_entity_ids\n")
        f995.write("source1_entity_id\tmatched_entity_ids\n")
        
        s1_entities_099 = 0
        s1_entities_0995 = 0
        
        for s1_id in s1_df['entity_id']:
            m99 = matches_099.get(s1_id, [])
            m995 = matches_0995.get(s1_id, [])
            
            # Remove duplicates while preserving order
            m99_unique = []
            seen99 = set()
            for m in m99:
                if m not in seen99:
                    seen99.add(m)
                    m99_unique.append(m)
                    
            m995_unique = []
            seen995 = set()
            for m in m995:
                if m not in seen995:
                    seen995.add(m)
                    m995_unique.append(m)
                    
            f99.write(f"{s1_id}\t{','.join(m99_unique)}\n")
            f995.write(f"{s1_id}\t{','.join(m995_unique)}\n")
            
            if m99_unique: s1_entities_099 += 1
            if m995_unique: s1_entities_0995 += 1

    total_time = time.time() - t_start
    print(f"\n--- Metrics ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"0.99 Matches: S1 with matches = {s1_entities_099}, Empty S1 = {len(s1_df) - s1_entities_099}")
    print(f"0.995 Matches: S1 with matches = {s1_entities_0995}, Empty S1 = {len(s1_df) - s1_entities_0995}")


if __name__ == "__main__":
    run_threshold_inference()
