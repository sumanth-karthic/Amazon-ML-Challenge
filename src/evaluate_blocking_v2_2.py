import os
import sys
import time
import psutil
import pandas as pd
import tracemalloc

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_chunk as preprocess_pipeline
from blocking_v2_2 import BlockingStrategiesV2_2

def load_benchmark_data(base_dir):
    # Same deterministic sampling as V2.1
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep='\t')
    
    print("Loading Source 1 (50k deterministically sampled)...")
    s1_df = pd.read_csv(s1_path, sep='\t')
    s1_df = s1_df.sort_values('entity_id').head(50000)
    s1_df['address_missing'] = s1_df['business_address'].isna() | (s1_df['business_address'] == "")
    s1_df = preprocess_pipeline(s1_df)
    
    s1_entities = set(s1_df['entity_id'])
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_entities)]
    
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

    print(f"Loading Source 2 (Target size {s2s3_rows} + required true matches)...")
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s2_true = {x for x in true_matches if x.startswith('S2-')}
    s2_df = load_source(s2_path, s2_true)
    
    print(f"Loading Source 3 (Target size {s2s3_rows} + required true matches)...")
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    s3_true = {x for x in true_matches if x.startswith('S3-')}
    s3_df = load_source(s3_path, s3_true)
    
    return s1_df, s2_df, s3_df, gt_df

def print_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_df, s2_df, s3_df, gt_df = load_benchmark_data(base_dir)
    
    print("Building blocking index V2.2 (SciPy Sparse Matrix)...")
    tracemalloc.start()
    t0 = time.time()
    
    # max_postings=500 is used to mimic the candidate limitation of V2.1
    blocker = BlockingStrategiesV2_2(max_postings=500, num_ngrams=5)
    
    print(f"Fitting frequencies (RAM: {print_memory():.1f} MB)...")
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    
    print(f"Building indexes (RAM: {print_memory():.1f} MB)...")
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    
    print(f"Finalizing indexes (RAM: {print_memory():.1f} MB)...")
    blocker.finalize_indexes()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Index built in {time.time()-t0:.2f}s")
    print(f"Peak tracemalloc memory usage: {peak_mem / 10**6:.2f} MB")
    print(f"Current Process RAM: {print_memory():.1f} MB")
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids']).split(',')
        gt_dict[row['source1_entity_id']] = set(matches)
        
    k_values = [50, 100, 150, 250, 300, 500]
    
    print("Evaluating strategies...")
    t_eval_start = time.time()
    
    total_retrieval_and_scoring_time = 0
    all_candidates_info = []
    
    rows = list(s1_df.itertuples(index=False))
    k_max = max(k_values)
    
    for i, row in enumerate(rows):
        if i > 0 and i % 5000 == 0:
            print(f"Processed {i}/50000 queries (RAM: {print_memory():.1f} MB)...", flush=True)
            
        s1_id = getattr(row, 'entity_id')
        addr_missing = getattr(row, 'address_missing', False)
        
        tr0 = time.time()
        scored_cands = blocker.retrieve_and_score(row, k=k_max)
        tr1 = time.time()
        
        total_retrieval_and_scoring_time += (tr1 - tr0)
        all_candidates_info.append((s1_id, scored_cands, addr_missing))
        
    print(f"Total retrieval and scoring time: {total_retrieval_and_scoring_time:.2f}s")
    
    report = []
    for k in k_values:
        total_true = 0
        total_s2_true = 0
        total_s3_true = 0
        total_both_true = 0
        missing_addr_true = 0
        
        possible_true = 0
        possible_s2 = 0
        possible_s3 = 0
        possible_both = 0
        possible_missing_addr = 0
        
        total_generated = 0
        cands_per_s1 = []
        
        for s1_id, scored_cands, addr_missing in all_candidates_info:
            top_cands = scored_cands[:k]
            total_generated += len(top_cands)
            cands_per_s1.append(len(top_cands))
            
            if s1_id in gt_dict:
                true_cands = gt_dict[s1_id]
                has_s2 = any(c.startswith('S2-') for c in true_cands)
                has_s3 = any(c.startswith('S3-') for c in true_cands)
                
                possible_true += len(true_cands)
                if has_s2: possible_s2 += 1
                if has_s3: possible_s3 += 1
                if has_s2 and has_s3: possible_both += 1
                if addr_missing: possible_missing_addr += len(true_cands)
                
                s2_found = 0
                s3_found = 0
                for cand_id, _ in top_cands:
                    if cand_id in true_cands:
                        total_true += 1
                        if addr_missing:
                            missing_addr_true += 1
                        if cand_id.startswith('S2-'):
                            s2_found = 1
                        if cand_id.startswith('S3-'):
                            s3_found = 1
                            
                total_s2_true += s2_found
                total_s3_true += s3_found
                if s2_found and s3_found:
                    total_both_true += 1
                    
        cands_series = pd.Series(cands_per_s1)
        avg_cands = cands_series.mean()
        p95 = cands_series.quantile(0.95)
        p99 = cands_series.quantile(0.99)
        
        recall = total_true / possible_true if possible_true else 0
        s2_recall = total_s2_true / possible_s2 if possible_s2 else 0
        s3_recall = total_s3_true / possible_s3 if possible_s3 else 0
        both_recall = total_both_true / possible_both if possible_both else 0
        missing_recall = missing_addr_true / possible_missing_addr if possible_missing_addr else 0
        
        red_ratio = 1.0 - (total_generated / (len(s1_df) * (len(s2_df) + len(s3_df))))
        
        report.append({
            'k': k,
            'recall': recall,
            's2_recall': s2_recall,
            's3_recall': s3_recall,
            'both_recall': both_recall,
            'missing_addr_recall': missing_recall,
            'avg_cands': avg_cands,
            'p95': p95,
            'p99': p99,
            'total_generated': total_generated,
            'reduction_ratio': red_ratio
        })
        
    print("\nBLOCKING BENCHMARK REPORT V2.2 (Sparse Matrix)")
    print("=" * 80)
    print(f"S1 Entities: {len(s1_df)}")
    print(f"S2 Candidate Pool: {len(s2_df)}")
    print(f"S3 Candidate Pool: {len(s3_df)}")
    print(f"Peak Tracemalloc Memory (Index): {peak_mem / 10**6:.2f} MB")
    print(f"Process RAM limit context: {print_memory():.2f} MB")
    print(f"Total end-to-end evaluation time: {time.time() - t_eval_start:.2f}s")
    print()
    print("K    | Recall | S2 Rec | S3 Rec | Both Rec | MissAddr | Avg Cand | P95 | P99 ")
    print("-" * 80)
    for r in report:
        print(f"{r['k']:<4} | {r['recall']:.4f} | {r['s2_recall']:.4f} | {r['s3_recall']:.4f} | "
              f"{r['both_recall']:.4f}   | {r['missing_addr_recall']:.4f}   | {r['avg_cands']:<8.1f} | "
              f"{r['p95']:<3.0f} | {r['p99']:<3.0f}")
              
    report_path = os.path.join(base_dir, 'reports', 'blocking_experiment_v2_2.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BLOCKING BENCHMARK REPORT V2.2 (Sparse Matrix)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"S1 Entities: {len(s1_df)}\n")
        f.write(f"S2 Candidate Pool: {len(s2_df)}\n")
        f.write(f"S3 Candidate Pool: {len(s3_df)}\n\n")
        f.write(f"Peak Memory (Index): {peak_mem / 10**6:.2f} MB\n\n")
        f.write("K    | Recall | Avg Cand | P95 | P99 | Reduction\n")
        f.write("-" * 80 + "\n")
        for r in report:
            f.write(f"{r['k']:<4} | {r['recall']:.4f} | {r['avg_cands']:<8.1f} | {r['p95']:<3.0f} | {r['p99']:<3.0f} | {r['reduction_ratio']:.6f}\n")
            
        r50 = next(r for r in report if r['k'] == 50)
        f.write(f"\nDetailed Breakdown (K=50):\n")
        f.write(f"- Total candidate pairs generated: {r50['total_generated']}\n")
        f.write(f"- Recall for S2 matches: {r50['s2_recall']:.4f}\n")
        f.write(f"- Recall for S3 matches: {r50['s3_recall']:.4f}\n")
        f.write(f"- Recall for entities matching BOTH: {r50['both_recall']:.4f}\n")
        f.write(f"- Recall for entities with missing addresses: {r50['missing_addr_recall']:.4f}\n")

if __name__ == "__main__":
    evaluate()
