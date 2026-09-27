import os
import sys
import time
import psutil
import tracemalloc
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_chunk as preprocess_pipeline
from blocking_v2_2 import BlockingStrategiesV2_2
from matching_features import PairFeatureGenerator

def print_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_data(base_dir):
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    
    print("Loading Source 1 (50k)...")
    s1_df = pd.read_csv(s1_path, sep='\t')
    s1_df = s1_df.sort_values('entity_id').head(50000)
    s1_df['address_missing'] = s1_df['business_address'].isna() | (s1_df['business_address'] == "")
    s1_df = preprocess_pipeline(s1_df)
    
    s1_entities = set(s1_df['entity_id'])
    
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_entities)]
    
    gt_dict = {}
    true_matches_all = set()
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids']).split(',')
        s1_id = row['source1_entity_id']
        gt_dict[s1_id] = set(matches)
        true_matches_all.update(matches)
        
    s2s3_rows = 500000
    
    def load_source(path, required_ids):
        df_list = []
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=100000)
        remaining = s2s3_rows
        for chunk in chunk_iter:
            chunk['address_missing'] = chunk['business_address'].isna() | (chunk['business_address'] == "")
            chunk = preprocess_pipeline(chunk)
            if remaining > 0:
                keep = chunk.head(remaining)
                df_list.append(keep)
                remaining -= len(keep)
            
            req_chunk = chunk[chunk['entity_id'].isin(required_ids)]
            if not req_chunk.empty:
                df_list.append(req_chunk)
                
        return pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['entity_id'])

    print(f"Loading Source 2 (Target size {s2s3_rows})...")
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s2_true = {x for x in true_matches_all if x.startswith('S2-')}
    s2_df = load_source(s2_path, s2_true)
    
    print(f"Loading Source 3 (Target size {s2s3_rows})...")
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    s3_true = {x for x in true_matches_all if x.startswith('S3-')}
    s3_df = load_source(s3_path, s3_true)
    
    return s1_df, s2_df, s3_df, gt_dict

def generate_features():
    tracemalloc.start()
    t_start = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_df, s2_df, s3_df, gt_dict = load_data(base_dir)
    
    print(f"Data loaded. Peak RAM: {print_memory():.1f} MB")
    
    print("Building V2.2 Sparse Index...")
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    blocker.finalize_indexes()
    
    idf_dict = {term: blocker.idf.get(idx, 0.5) for term, idx in blocker.vocab.items()}
    generator = PairFeatureGenerator(idf_dict)
    
    print("Building S2/S3 Metadata Dictionary for fast lookup...")
    s2s3_store = {}
    for df in [s2_df, s3_df]:
        cols = df.columns.tolist()
        for row in df.itertuples(index=False):
            s2s3_store[row.entity_id] = dict(zip(cols, row))
            
    print(f"Dictionary built. RAM: {print_memory():.1f} MB")
    
    out_path = os.path.join(base_dir, "dataset", "train_features.parquet")
    if os.path.exists(out_path):
        os.remove(out_path)
        
    print("Generating candidate pairs and features...")
    s1_cols = s1_df.columns.tolist()
    s1_rows = list(s1_df.itertuples(index=False))
    
    total_pairs = 0
    total_positive = 0
    total_negative = 0
    s1_with_positive = 0
    
    total_true_matches_possible = sum(len(v) for v in gt_dict.values())
    true_matches_hit = 0
    
    chunk_size = 200000
    current_chunk = {k: [] for k in ['s1_entity_id', 'cand_entity_id', 'label']}
    
    parquet_writer = None
    
    t_gen_start = time.time()
    
    # We need a reference row to know all keys to initialize `current_chunk` correctly,
    # but we will just populate them on the fly
    initialized_keys = False
    
    for i, s1_row in enumerate(s1_rows):
        if i > 0 and i % 2500 == 0:
            print(f"Processed {i}/50000 S1 entities. Pairs: {total_pairs}. RAM: {print_memory():.1f} MB")
            
        s1_id = s1_row.entity_id
        s1_meta = dict(zip(s1_cols, s1_row))
        
        cands = blocker.retrieve_and_score(s1_row, k=50)
        true_cands = gt_dict.get(s1_id, set())
        
        has_pos = False
        
        for rank, (cand_id, score) in enumerate(cands, 1):
            if cand_id in true_cands:
                label = 1
                total_positive += 1
                true_matches_hit += 1
                has_pos = True
            else:
                label = 0
                total_negative += 1
                
            s2_meta = s2s3_store[cand_id]
            
            feat = generator.generate_features(s1_meta, s2_meta, blocking_score=score, blocking_rank=rank)
            
            if not initialized_keys:
                for k in feat.keys():
                    if k not in current_chunk:
                        current_chunk[k] = []
                initialized_keys = True
                
            current_chunk['s1_entity_id'].append(s1_id)
            current_chunk['cand_entity_id'].append(cand_id)
            current_chunk['label'].append(label)
            for k, v in feat.items():
                current_chunk[k].append(v)
                
            total_pairs += 1
            
            if len(current_chunk['s1_entity_id']) >= chunk_size:
                df_chunk = pd.DataFrame(current_chunk)
                table = pa.Table.from_pandas(df_chunk)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                parquet_writer.write_table(table)
                current_chunk = {k: [] for k in current_chunk.keys()}
                print(f"Flushed chunk to Parquet. RAM: {print_memory():.1f} MB")
                
        if has_pos:
            s1_with_positive += 1
            
    # Write remaining
    if current_chunk['s1_entity_id']:
        df_chunk = pd.DataFrame(current_chunk)
        table = pa.Table.from_pandas(df_chunk)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
        parquet_writer.write_table(table)
        
    if parquet_writer:
        parquet_writer.close()
        
    t_end = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    
    missing_true = total_true_matches_possible - true_matches_hit
    pos_rate = total_positive / total_pairs if total_pairs > 0 else 0
    
    print("\n" + "="*80)
    print("TRAINING FEATURE GENERATION REPORT")
    print("="*80)
    print(f"Total Candidate Pairs:      {total_pairs:,}")
    print(f"Positive Pairs (Label=1):   {total_positive:,}")
    print(f"Negative Pairs (Label=0):   {total_negative:,}")
    print(f"Positive Rate:              {pos_rate:.4%}")
    print(f"S1 Entities w/ >=1 Pos:     {s1_with_positive:,} / 50,000")
    print(f"True Matches Hit:           {true_matches_hit:,} / {total_true_matches_possible:,}")
    print(f"True Matches Missing:       {missing_true:,} (Recall loss from Candidate Gen)")
    print(f"Peak RAM:                   {peak_mem / 10**6:.1f} MB")
    print(f"Total Runtime:              {t_end - t_start:.2f} seconds")
    print(f"Feature Gen Only Time:      {t_end - t_gen_start:.2f} seconds")
    print(f"Parquet File Size:          {file_size_mb:.2f} MB")
    print("="*80)
    print(f"Saved dataset to: {out_path}")

if __name__ == "__main__":
    generate_features()
