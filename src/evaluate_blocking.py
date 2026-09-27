import os
import sys
import pandas as pd
import numpy as np
import ast
import time
from preprocessing import preprocess_chunk
from blocking import BlockingStrategies

def load_benchmark_data(base_dir, s1_rows=50000, s2s3_rows=500000):
    print("Loading datasets...")
    # Load 50K S1
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep='\t', nrows=s1_rows, dtype=str, quoting=3, on_bad_lines='warn')
    s1_df = preprocess_chunk(s1_df)
    
    # Load ground truth for these 50K S1
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
    
    # Filter ground truth to only include the sampled S1
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_df['entity_id'])]
    
    # Find all true S2 and S3 IDs we must include in our candidate index
    true_matches = set()
    for matches_str in gt_df['matched_entity_ids'].dropna():
        if matches_str.strip():
            true_matches.update([x.strip() for x in matches_str.split(',')])
            
    # Function to load S2/S3 ensuring true matches are included
    def load_source(path, true_matches_set):
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=100000, dtype=str, quoting=3, on_bad_lines='warn')
        df_list = []
        rows_loaded = 0
        missing_targets = true_matches_set.copy()
        
        for chunk in chunk_iter:
            # We want to keep records that are either in missing_targets OR we haven't reached the row limit
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

    print(f"Loading Source 2 (Target size {s2s3_rows} + required true matches)...")
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s2_true = {x for x in true_matches if x.startswith('S2-')}
    s2_df = load_source(s2_path, s2_true)
    
    print(f"Loading Source 3 (Target size {s2s3_rows} + required true matches)...")
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    s3_true = {x for x in true_matches if x.startswith('S3-')}
    s3_df = load_source(s3_path, s3_true)
    
    return s1_df, s2_df, s3_df, gt_df

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_df, s2_df, s3_df, gt_df = load_benchmark_data(base_dir)
    
    print("Building blocking index...")
    t0 = time.time()
    blocker = BlockingStrategies(max_postings=5000) # Configurable max postings
    blocker.fit_dataframe(s2_df)
    blocker.fit_dataframe(s3_df)
    print(f"Index built in {time.time()-t0:.2f}s")
    
    # Ground truth mapping
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matches).split(','))
            
    strategies = {
        'A. Exact name': blocker.get_candidates_A,
        'B. Name + country': blocker.get_candidates_B,
        'C. Rare name token + country': blocker.get_candidates_C,
        'D. Postal + name token': blocker.get_candidates_D,
        'E. Name token + addr token': blocker.get_candidates_E,
        'F. Union (A+B+C+D)': blocker.get_candidates_F,
    }
    
    report = []
    
    print("Evaluating strategies...")
    for s_name, strategy_func in strategies.items():
        total_true = 0
        total_found = 0
        cand_counts = []
        
        # Breakdown counters for Strategy F
        s2_true = 0
        s3_true = 0
        both_true = 0
        s2_found = 0
        s3_found = 0
        both_found = 0
        missing_addr_true = 0
        missing_addr_found = 0
        
        for _, row in s1_df.iterrows():
            s1_id = row['entity_id']
            true_set = gt_dict.get(s1_id, set())
            
            candidates = strategy_func(row)
            
            found = candidates.intersection(true_set)
            
            total_true += len(true_set)
            total_found += len(found)
            cand_counts.append(len(candidates))
            
            if s_name.startswith('F'):
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
                    
                if row['address_missing']:
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
            'recall': recall,
            'avg_cand': avg_cand,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'max_cand': max_c,
            'total_cands': total_c,
            'reduction_ratio': reduction,
            'singleton_rate': singletons / len(cand_counts)
        })
        
        if s_name.startswith('F'):
            f_stats = {
                's2_recall': s2_found / s2_true if s2_true > 0 else 1.0,
                's3_recall': s3_found / s3_true if s3_true > 0 else 1.0,
                'both_recall': both_found / both_true if both_true > 0 else 1.0,
                'missing_addr_recall': missing_addr_found / missing_addr_true if missing_addr_true > 0 else 1.0
            }
            
            # Save candidates for F
            print("Saving benchmark output for Strategy F...")
            out_path = os.path.join(base_dir, "output", "candidate_pairs.tsv")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("source1_entity_id\tcandidate_entity_ids\n")
                for _, row in s1_df.iterrows():
                    s1_id = row['entity_id']
                    cands = strategy_func(row)
                    out_f.write(f"{s1_id}\t{','.join(cands)}\n")
                    
    # Generate Report
    rep_path = os.path.join(base_dir, "reports", "blocking_experiment.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("BLOCKING BENCHMARK REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"S1 Entities: {len(s1_df)}\n")
        f.write(f"S2 Candidate Pool: {len(s2_df)}\n")
        f.write(f"S3 Candidate Pool: {len(s3_df)}\n\n")
        
        # Write table
        header = f"{'Strategy':<30} | {'Recall':<8} | {'Avg Cand':<8} | {'P95 Cand':<8} | {'P99 Cand':<8} | {'Max Cand':<8} | {'Reduction'}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in report:
            f.write(f"{r['strategy']:<30} | {r['recall']:.4f} | {r['avg_cand']:<8.1f} | {r['p95']:<8.0f} | {r['p99']:<8.0f} | {r['max_cand']:<8.0f} | {r['reduction_ratio']:.6f}\n")
            
        f.write("\nDetailed Breakdown for Strategy F (Union):\n")
        f.write(f"- Median candidates (P50): {report[-1]['p50']}\n")
        f.write(f"- P90 candidates: {report[-1]['p90']}\n")
        f.write(f"- Total candidate pairs generated: {report[-1]['total_cands']}\n")
        f.write(f"- Singleton candidate rate (no candidates found): {report[-1]['singleton_rate']*100:.2f}%\n")
        f.write(f"- Recall for S2 matches: {f_stats['s2_recall']:.4f}\n")
        f.write(f"- Recall for S3 matches: {f_stats['s3_recall']:.4f}\n")
        f.write(f"- Recall for entities matching BOTH: {f_stats['both_recall']:.4f}\n")
        f.write(f"- Recall for entities with missing addresses: {f_stats['missing_addr_recall']:.4f}\n")
        
    print(f"Report written to {rep_path}")

if __name__ == "__main__":
    evaluate()
