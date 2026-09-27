import os
import sys
import pandas as pd
import numpy as np
import time
import tracemalloc
from preprocessing import preprocess_chunk
from blocking_v2_1 import BlockingStrategiesV2_1

def load_benchmark_data(base_dir, s1_rows=50000, s2s3_rows=500000):
    print("Loading datasets...")
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep='\t', nrows=s1_rows, dtype=str, quoting=3, on_bad_lines='warn')
    s1_df = preprocess_chunk(s1_df)
    
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_df['entity_id'])]
    
    true_matches = set()
    for matches_str in gt_df['matched_entity_ids'].dropna():
        if matches_str.strip():
            true_matches.update([x.strip() for x in matches_str.split(',')])
            
    def load_source(path, true_matches_set):
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=100000, dtype=str, quoting=3, on_bad_lines='warn')
        df_list = []
        rows_loaded = 0
        missing_targets = true_matches_set.copy()
        
        for chunk in chunk_iter:
            if rows_loaded < s2s3_rows:
                keep = chunk
                rows_loaded += len(chunk)
            else:
                keep = chunk[chunk['entity_id'].isin(missing_targets)]
                
            if len(keep) > 0:
                missing_targets.difference_update(keep['entity_id'])
                df_list.append(preprocess_chunk(keep))
                
            if rows_loaded >= s2s3_rows and len(missing_targets) == 0:
                break
                
        return pd.concat(df_list, ignore_index=True)

    print(f"Loading Source 2 (Target size {s2s3_rows} + required true matches)...", flush=True)
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s2_true = {x for x in true_matches if x.startswith('S2-')}
    s2_df = load_source(s2_path, s2_true)
    
    print(f"Loading Source 3 (Target size {s2s3_rows} + required true matches)...", flush=True)
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    s3_true = {x for x in true_matches if x.startswith('S3-')}
    s3_df = load_source(s3_path, s3_true)
    
    return s1_df, s2_df, s3_df, gt_df

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_df, s2_df, s3_df, gt_df = load_benchmark_data(base_dir)
    
    print("Building blocking index V2.1...", flush=True)
    tracemalloc.start()
    t0 = time.time()
    
    # Using 5 ngrams as requested, max_postings=500
    blocker = BlockingStrategiesV2_1(max_postings=500, num_ngrams=5)
    
    print("Fitting frequencies...", flush=True)
    blocker.fit_frequencies(s2_df)
    blocker.fit_frequencies(s3_df)
    
    print("Building indexes...", flush=True)
    blocker.build_indexes(s2_df)
    blocker.build_indexes(s3_df)
    
    print("Finalizing indexes...", flush=True)
    blocker.finalize_indexes()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Index built in {time.time()-t0:.2f}s", flush=True)
    print(f"Peak memory usage for index: {peak_mem / 10**6:.2f} MB", flush=True)
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matches).split(','))
            
    k_values = [10, 20, 30, 50, 75, 100]
    
    # We will test the Combined strategy
    strategy_configs = {
        'Combined V2.1': ['A', 'B', 'C', 'D', 'E']
    }
    
    report = []
    
    print("Evaluating strategies...")
    for s_name, strat_keys in strategy_configs.items():
        print(f"Running strategy: {s_name} (single threaded)")
        
        total_retrieval_time = 0
        total_scoring_time = 0
        all_candidates_info = []
        
        rows = list(s1_df.itertuples(index=False))
        k_max = max(k_values)
        
        for i, row in enumerate(rows):
            if i > 0 and i % 5000 == 0:
                print(f"Processed {i}/50000 rows...")
                
            s1_id = getattr(row, 'entity_id')
            addr_missing = getattr(row, 'address_missing', False)
            
            tr0 = time.time()
            candidates = blocker.retrieve_candidates(row, strategies=strat_keys)
            tr1 = time.time()
            
            if candidates:
                ts0 = time.time()
                scored_cands = blocker.rank_and_topk(row, candidates, k=k_max)
                ts1 = time.time()
            else:
                scored_cands = []
                ts0, ts1 = 0, 0
                
            total_retrieval_time += (tr1 - tr0)
            total_scoring_time += (ts1 - ts0)
            all_candidates_info.append((s1_id, scored_cands, addr_missing))
            
        print(f"Total retrieval time: {total_retrieval_time:.2f}s")
        print(f"Total scoring time: {total_scoring_time:.2f}s")
            
        for k in k_values:
            total_true = 0
            total_found = 0
            cand_counts = []
            
            s2_true, s3_true, both_true = 0, 0, 0
            s2_found, s3_found, both_found = 0, 0, 0
            missing_addr_true, missing_addr_found = 0, 0
            
            for s1_id, scored_cands, addr_missing in all_candidates_info:
                true_set = gt_dict.get(s1_id, set())
                top_k_candidates = set(scored_cands[:k])
                
                found = top_k_candidates.intersection(true_set)
                
                total_true += len(true_set)
                total_found += len(found)
                cand_counts.append(len(top_k_candidates))
                
                t_s2 = {x for x in true_set if x.startswith('S2-')}
                t_s3 = {x for x in true_set if x.startswith('S3-')}
                f_s2 = {x for x in found if x.startswith('S2-')}
                f_s3 = {x for x in found if x.startswith('S3-')}
                
                s2_true += len(t_s2)
                s3_true += len(t_s3)
                s2_found += len(f_s2)
                s3_found += len(f_s3)
                if len(t_s2) > 0 and len(t_s3) > 0:
                    both_true += len(true_set)
                    both_found += len(found)
                    
                if addr_missing:
                    missing_addr_true += len(true_set)
                    missing_addr_found += len(found)
            
            recall = total_found / total_true if total_true > 0 else 1.0
            avg_cand = np.mean(cand_counts)
            p50 = np.percentile(cand_counts, 50)
            p90 = np.percentile(cand_counts, 90)
            p95 = np.percentile(cand_counts, 95)
            p99 = np.percentile(cand_counts, 99)
            max_c = np.max(cand_counts)
            total_c = np.sum(cand_counts)
            singletons = sum(1 for c in cand_counts if c == 0)
            reduction = 1.0 - (total_c / (len(s1_df) * (len(s2_df) + len(s3_df))))
            
            report.append({
                'strategy': s_name,
                'k': k,
                'recall': recall,
                'avg_cand': avg_cand,
                'p50': p50,
                'p90': p90,
                'p95': p95,
                'p99': p99,
                'max_cand': max_c,
                'total_cands': total_c,
                'reduction_ratio': reduction,
                'singleton_rate': singletons / len(cand_counts),
                's2_recall': s2_found / s2_true if s2_true > 0 else 1.0,
                's3_recall': s3_found / s3_true if s3_true > 0 else 1.0,
                'both_recall': both_found / both_true if both_true > 0 else 1.0,
                'missing_addr_recall': missing_addr_found / missing_addr_true if missing_addr_true > 0 else 1.0
            })
                    
    rep_path = os.path.join(base_dir, "reports", "blocking_experiment_v2.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("BLOCKING BENCHMARK REPORT V2.1\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"S1 Entities: {len(s1_df)}\n")
        f.write(f"S2 Candidate Pool: {len(s2_df)}\n")
        f.write(f"S3 Candidate Pool: {len(s3_df)}\n\n")
        f.write(f"Peak Memory (Index): {peak_mem / 10**6:.2f} MB\n\n")
        
        header = f"{'Strategy':<30} | {'K':<4} | {'Recall':<8} | {'Avg Cand':<8} | {'P95':<5} | {'P99':<5} | {'Reduction'}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in report:
            f.write(f"{r['strategy']:<30} | {r['k']:<4} | {r['recall']:.4f} | {r['avg_cand']:<8.1f} | {r['p95']:<5.0f} | {r['p99']:<5.0f} | {r['reduction_ratio']:.6f}\n")
            
        f.write("\nDetailed Breakdown for Combined V2.1 (K=50):\n")
        combined_50 = next(r for r in report if r['k'] == 50)
        f.write(f"- Median candidates (P50): {combined_50['p50']}\n")
        f.write(f"- Total candidate pairs generated: {combined_50['total_cands']}\n")
        f.write(f"- Recall for S2 matches: {combined_50['s2_recall']:.4f}\n")
        f.write(f"- Recall for S3 matches: {combined_50['s3_recall']:.4f}\n")
        f.write(f"- Recall for entities matching BOTH: {combined_50['both_recall']:.4f}\n")
        f.write(f"- Recall for entities with missing addresses: {combined_50['missing_addr_recall']:.4f}\n")

    print(f"Report written to {rep_path}")

if __name__ == "__main__":
    evaluate()
