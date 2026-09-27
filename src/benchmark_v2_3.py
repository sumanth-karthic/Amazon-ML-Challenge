import pandas as pd
import numpy as np
import time
import os
import psutil

from evaluate_blocking_v2_1 import load_benchmark_data
from blocking_v2_2 import BlockingStrategiesV2_2
from blocking_v2_3 import BlockingStrategiesV2_3
import collections

def print_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory: {process.memory_info().rss / 1024**2:.2f} MB")

def run_benchmark(blocker, name, s1_eval, s2_eval, s3_eval, gt_eval, k_values=[50, 100, 150]):
    print(f"\n{'='*80}\nBENCHMARKING: {name}\n{'='*80}")
    
    t0 = time.time()
    blocker.fit_frequencies(s2_eval)
    blocker.fit_frequencies(s3_eval)
    
    blocker.build_indexes(s2_eval)
    blocker.build_indexes(s3_eval)
    
    blocker.finalize_indexes()
    t1 = time.time()
    
    print(f"Index build time: {t1 - t0:.2f}s")
    print_memory()
    
    chunk_size = 5000
    s1_rows = list(s1_eval.itertuples(index=False))
    
    for k in k_values:
        print(f"\n--- Evaluating K={k} ---")
        t2 = time.time()
        
        all_cands = {}
        lens = []
        for i in range(0, len(s1_rows), chunk_size):
            chunk = s1_rows[i : i + chunk_size]
            batch_results = blocker.retrieve_and_score_batch(chunk, k=k)
            for j, res in enumerate(batch_results):
                e_id = getattr(chunk[j], 'entity_id')
                # res is list of (cand_id, score)
                cands = [c[0] for c in res]
                all_cands[e_id] = cands
                lens.append(len(cands))
                
        t3 = time.time()
        print(f"Retrieval time: {t3 - t2:.2f}s")
        print(f"Avg candidates: {np.mean(lens):.2f}, Median: {np.median(lens):.2f}, Max: {np.max(lens)}")
        
        
        # Evaluate recall
        gt_total = 0
        s2_hits = 0
        s3_hits = 0
        s2_total = 0
        s3_total = 0
        both_hits = 0
        both_total = 0
        
        # Missing address
        missing_addr_s1 = s1_eval[s1_eval['business_address'].isna()]['entity_id']
        missing_hits = 0
        missing_total = 0
        
        country_stats = collections.defaultdict(lambda: {'hits': 0, 'total': 0})
        
        for idx, row in gt_eval.iterrows():
            s1 = row['source1_entity_id']
            matches_str = row['matched_entity_ids']
            if pd.isna(matches_str) or not str(matches_str).strip():
                continue
                
            targets = [x.strip() for x in str(matches_str).split(',')]
            cands = set(all_cands.get(s1, []))
            
            s2_t = [t for t in targets if t.startswith('S2')]
            s3_t = [t for t in targets if t.startswith('S3')]
            
            gt_total += len(targets)
            
            hit_s2 = sum(1 for t in s2_t if t in cands)
            hit_s3 = sum(1 for t in s3_t if t in cands)
            
            s2_hits += hit_s2
            s3_hits += hit_s3
            s2_total += len(s2_t)
            s3_total += len(s3_t)
            
            if len(s2_t) > 0 and len(s3_t) > 0:
                both_total += 1
                if hit_s2 > 0 and hit_s3 > 0:
                    both_hits += 1
                    
            if s1 in missing_addr_s1.values:
                missing_total += 1
                if (hit_s2 + hit_s3) > 0:
                    missing_hits += 1
                    
            country = s1_eval[s1_eval['entity_id'] == s1]['country'].values
            c = str(country[0]).lower().strip() if len(country) > 0 else 'unknown'
            country_stats[c]['total'] += len(targets)
            country_stats[c]['hits'] += (hit_s2 + hit_s3)
            
        overall = (s2_hits + s3_hits) / gt_total if gt_total > 0 else 0
        print(f"Overall Recall: {overall:.4f} ({s2_hits+s3_hits}/{gt_total})")
        print(f"S2 Recall: {s2_hits/s2_total if s2_total > 0 else 0:.4f}")
        print(f"S3 Recall: {s3_hits/s3_total if s3_total > 0 else 0:.4f}")
        print(f"Both-Source Recall (entity level): {both_hits/both_total if both_total > 0 else 0:.4f}")
        print(f"Missing-Address Recall: {missing_hits/missing_total if missing_total > 0 else 0:.4f}")
        
        print("Recall by Country:")
        for c, stats in country_stats.items():
            r = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {c}: {r:.4f} ({stats['hits']}/{stats['total']})")
        
    return all_cands

def main():
    base_dir = "../"
    print("Loading benchmark data...")
    s1_eval, s2_eval, s3_eval, gt_eval = load_benchmark_data(base_dir, s1_rows=50000, s2s3_rows=500000)
    
    # We want to specifically check some ground truth matches with missing addresses
    # However, s1_eval has 0 missing addresses. Let's make sure we find some missing address targets in s2/s3.
    missing_addr_s2s3_gt = gt_eval.copy()
    
    # We will just evaluate both blockers
    blocker22 = BlockingStrategiesV2_2(max_postings=10000, num_ngrams=5)
    cands22 = run_benchmark(blocker22, "V2.2 (Words)", s1_eval, s2_eval, s3_eval, gt_eval)
    
    blocker23 = BlockingStrategiesV2_3(max_postings=10000, name_ngram_range=(2,4))
    cands23 = run_benchmark(blocker23, "V2.3 (Char-WB Ngrams)", s1_eval, s2_eval, s3_eval, gt_eval)
    
    print(f"\n{'='*80}\nSANITY CHECK: MISSING ADDRESSES\n{'='*80}")
    # Find some ground truth pairs where one side has a missing address
    # Since S1 has no missing addresses in the first 50k, let's look at S2/S3
    s2s3_eval = pd.concat([s2_eval, s3_eval]).set_index('entity_id')
    
    missing_count = 0
    for idx, row in gt_eval.iterrows():
        s1 = row['source1_entity_id']
        targets = str(row['matched_entity_ids']).split('|')
        
        for t in targets:
            if t in s2s3_eval.index:
                t_addr = s2s3_eval.loc[t, 'business_address']
                if pd.isna(t_addr) or str(t_addr).strip() == '':
                    missing_count += 1
                    s1_name = s1_eval[s1_eval['entity_id'] == s1]['business_name'].values[0]
                    t_name = s2s3_eval.loc[t, 'business_name']
                    t_country = s2s3_eval.loc[t, 'country']
                    
                    c22 = t in cands22.get(s1, [])
                    c23 = t in cands23.get(s1, [])
                    
                    print(f"S1: {s1} | S2/S3: {t}")
                    print(f"S1 Name: {s1_name}")
                    print(f"Target Name: {t_name}, Country: {t_country}")
                    print(f"Retrieved by V2.2: {c22} | Retrieved by V2.3: {c23}")
                    print("-" * 40)
                    
                    if missing_count >= 10:
                        break
        if missing_count >= 10:
            break
            
if __name__ == "__main__":
    main()
