import pandas as pd
import numpy as np
import time
import os
import gc
import scipy.sparse as sp
import lightgbm as lgb
import json
import shutil
import sys

from preprocessing import preprocess_chunk
from matching_features import PairFeatureGenerator
from blocking_v2_2 import BlockingStrategiesV2_2

CHUNK_SIZE = 500000
PROGRESS_FILE = 'progress_state.json'
PROGRESS_TMP = 'progress_state.tmp'
CKPT_099 = '../output/threshold_tests/checkpoint_0.99.tsv'
CKPT_0995 = '../output/threshold_tests/checkpoint_0.995.tsv'
CANDIDATE_FILE = '../candidate_pairs.tsv'

def run_threshold_inference(test_mode=False):
    print("Checking progress state...")
    start_chunk = 0
    start_rows = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
            start_chunk = state.get('completed_chunks', 0)
            start_rows = state.get('completed_rows', 0)
            print(f"Found checkpoint! Resuming from chunk {start_chunk} ({start_rows} rows).")
            
            # Verify file inconsistency
            if not os.path.exists(CKPT_099) or not os.path.exists(CKPT_0995):
                print("Inconsistency: progress file exists but checkpoints missing.")
                sys.exit(1)

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
    
    def score_pairs_fast(s1_df_chunk, cand_ids):
        # Find unique S1 rows to avoid duplicate string processing
        unique_s1 = s1_df_chunk.drop_duplicates(subset=['entity_id'])
        unique_rows = list(unique_s1.itertuples(index=False))
        
        q_data, q_row, q_col = [], [], []
        for i, s1_row in enumerate(unique_rows):
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
                    
        Q_unique = sp.csr_matrix((q_data, (q_row, q_col)), shape=(len(unique_rows), blocker.vocab_size), dtype=np.float32)
        
        # Map original s1_ids back to the unique Q matrix
        s1_to_idx = {eid: i for i, eid in enumerate(unique_s1['entity_id'])}
        row_indices = [s1_to_idx[eid] for eid in s1_df_chunk['entity_id']]
        Q_batch = Q_unique[row_indices, :]
        
        # Candidates matrix
        c_indices = [ent2idx[cid] for cid in cand_ids]
        C_batch = blocker.matrix[:, c_indices]
        
        # Element-wise multiply and sum
        scores = Q_batch.multiply(C_batch.T).sum(axis=1).A1
        return scores

    print("Loading LightGBM model...")
    model = lgb.Booster(model_file='../models/lgbm_model.txt')
    generator = PairFeatureGenerator(idf_dict=blocker.idf)
    feature_cols = model.feature_name()

    os.makedirs('../output/threshold_tests', exist_ok=True)
    
    if start_chunk == 0:
        with open(CKPT_099, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_id\n")
        with open(CKPT_0995, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_id\n")
    
    print("Streaming candidate pairs...")
    s1_df_indexed = s1_df.set_index('entity_id')
    
    file_stat = os.stat(CANDIDATE_FILE)
    
    col_names = ['source1_entity_id', 'candidate_entity_ids']
    if start_rows > 0:
        reader = pd.read_csv(CANDIDATE_FILE, sep='\t', names=col_names, skiprows=start_rows+1, chunksize=CHUNK_SIZE, dtype=str)
    else:
        reader = pd.read_csv(CANDIDATE_FILE, sep='\t', chunksize=CHUNK_SIZE, dtype=str)
    
    current_s1 = None
    current_rank = 0
    total_pairs = start_rows
    chunk_idx = start_chunk
    
    for chunk in reader:
        if start_rows > 0 and 'source1_entity_id' not in chunk.columns:
            chunk.columns = col_names
            
        s1_ids = chunk['source1_entity_id'].values
        cand_ids = chunk['candidate_entity_ids'].values
        
        s1_chunk_df = s1_df_indexed.loc[s1_ids].reset_index()
        s2_chunk_df = s2s3_df.loc[cand_ids].reset_index()
        
        scores = score_pairs_fast(s1_chunk_df, cand_ids)
        
        ranks = []
        for s1_id in s1_ids:
            if s1_id != current_s1:
                current_s1 = s1_id
                current_rank = 1
            else:
                current_rank += 1
            ranks.append(current_rank)
            
        feat_df = generator.generate_features_batch(s1_chunk_df, s2_chunk_df, scores, ranks)
        X_chunk = feat_df[feature_cols]
        
        preds = model.predict(X_chunk)
        
        mask_099 = preds >= 0.99
        mask_0995 = preds >= 0.995
        
        res_099 = []
        res_0995 = []
        for s1_id, cand_id in zip(s1_ids[mask_099], cand_ids[mask_099]):
            res_099.append(f"{s1_id}\t{cand_id}\n")
        for s1_id, cand_id in zip(s1_ids[mask_0995], cand_ids[mask_0995]):
            res_0995.append(f"{s1_id}\t{cand_id}\n")
            
        with open(CKPT_099, 'a', encoding='utf-8') as f:
            f.writelines(res_099)
        with open(CKPT_0995, 'a', encoding='utf-8') as f:
            f.writelines(res_0995)
            
        total_pairs += len(chunk)
        chunk_idx += 1
        
        state = {
            'completed_chunks': chunk_idx,
            'completed_rows': total_pairs,
            'source_size': file_stat.st_size,
            'source_mtime': file_stat.st_mtime,
            'thresholds': [0.99, 0.995],
            'timestamp': time.time()
        }
        with open(PROGRESS_TMP, 'w') as f:
            json.dump(state, f)
        os.replace(PROGRESS_TMP, PROGRESS_FILE)
        
        print(f"Processed chunk {chunk_idx} ({total_pairs} pairs)...")
        
        gc.collect()
        
        if test_mode and chunk_idx >= start_chunk + 2:
            print("Test mode: simulating interrupt after 2 chunks.")
            sys.exit(0)
            
    print("Inference fully completed! Aggregating final files...")
    for thr, ckpt_path, out_path in [(0.99, CKPT_099, '../output/threshold_tests/matching_results_0.99.tsv'), 
                                     (0.995, CKPT_0995, '../output/threshold_tests/matching_results_0.995.tsv')]:
        print(f"Aggregating {thr}...")
        df_ckpt = pd.read_csv(ckpt_path, sep='\t', dtype=str)
        
        grouped = df_ckpt.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(dict.fromkeys(x).keys())).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
        
        final_df = s1_df[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
        final_df = final_df.merge(grouped, on='source1_entity_id', how='left').fillna('')
        
        final_df.to_csv(out_path, sep='\t', index=False)
        print(f"Written {out_path} with {len(final_df)} rows.")

    print(f"All done! Total time: {time.time() - t_start:.2f}s")
    
if __name__ == '__main__':
    test_mode = '--test' in sys.argv
    run_threshold_inference(test_mode)
