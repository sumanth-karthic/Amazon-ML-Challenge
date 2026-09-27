import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import os
import sys

from preprocessing import preprocess_chunk
from blocking_v2_2 import BlockingStrategiesV2_2

# Setup fake test data
print("Loading small slice of dataset...")
test_dir = '../dataset/test'
s1_df = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', dtype=str, nrows=1000).fillna('')
s2_df = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', dtype=str, nrows=1000).fillna('')
s3_df = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', dtype=str, nrows=1000).fillna('')

s1_df = preprocess_chunk(s1_df)
s2_df = preprocess_chunk(s2_df)
s3_df = preprocess_chunk(s3_df)

s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)

print("Building V2.2 block index...")
blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
blocker.fit_frequencies(s2_df)
blocker.fit_frequencies(s3_df)
blocker.build_indexes(s2_df)
blocker.build_indexes(s3_df)
blocker.finalize_indexes()

ent2idx = {eid: i for i, eid in enumerate(blocker.entity_ids)}
s2s3_df.set_index('entity_id', inplace=True)

# Original function
def score_pairs_original(s1_rows, cand_ids):
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

# Optimized function
def score_pairs_optimized(s1_df_chunk, cand_ids):
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
    
    s1_to_idx = {eid: i for i, eid in enumerate(unique_s1['entity_id'])}
    row_indices = [s1_to_idx[eid] for eid in s1_df_chunk['entity_id']]
    Q_batch = Q_unique[row_indices, :]
    
    c_indices = [ent2idx[cid] for cid in cand_ids]
    C_batch = blocker.matrix[:, c_indices]
    
    scores = Q_batch.multiply(C_batch.T).sum(axis=1).A1
    return scores

print("Generating synthetic candidate pairs...")
s1_ids = s1_df['entity_id'].values[:500]
# Create duplicates to test grouping logic
s1_ids = np.repeat(s1_ids, 10) # 5000 pairs total, 10 cands per s1

cand_ids = np.random.choice(s2s3_df.index.values, size=len(s1_ids))

s1_chunk_df = s1_df.set_index('entity_id').loc[s1_ids].reset_index()
s1_rows_tuples = list(s1_chunk_df.itertuples(index=False))

print("Testing original...")
t0 = time.time()
scores_orig = score_pairs_original(s1_rows_tuples, cand_ids)
t_orig = time.time() - t0

print("Testing optimized...")
t0 = time.time()
scores_opt = score_pairs_optimized(s1_chunk_df, cand_ids)
t_opt = time.time() - t0

print(f"Original Time: {t_orig:.4f}s")
print(f"Optimized Time: {t_opt:.4f}s")

# Compare scores
diff = np.abs(scores_orig - scores_opt).max()
print(f"Max absolute difference in scores: {diff}")
if diff < 1e-6:
    print("EQUIVALENCE CHECK PASSED!")
else:
    print("EQUIVALENCE CHECK FAILED!")
