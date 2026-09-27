import os
import pandas as pd

def main():
    base_dir = "../"
    
    test_s1_path = os.path.join(base_dir, "dataset", "test", "test_source1.tsv")
    test_s2_path = os.path.join(base_dir, "dataset", "test", "test_source2.tsv")
    test_s3_path = os.path.join(base_dir, "dataset", "test", "test_source3.tsv")
    
    print("Loading test datasets...")
    s1 = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id', 'country'])
    s2 = pd.read_csv(test_s2_path, sep='\t', usecols=['entity_id', 'country'])
    s3 = pd.read_csv(test_s3_path, sep='\t', usecols=['entity_id', 'country'])
    
    def count_france(df):
        return sum(df['country'].str.lower().str.strip() == 'france')
        
    s1_france = count_france(s1)
    s2_france = count_france(s2)
    s3_france = count_france(s3)
    
    s1_total = len(s1)
    
    print(f"France S1 entities: {s1_france}")
    print(f"France S2 entities: {s2_france}")
    print(f"France S3 entities: {s3_france}")
    print(f"Percentage of test S1 that is France: {s1_france / s1_total * 100:.2f}%")
    
    # Load threshold results
    res_path = os.path.join(base_dir, "output", "matching_results.tsv")
    if os.path.exists(res_path):
        res = pd.read_csv(res_path, sep='\t')
        
        # Get France S1 ids
        france_s1_ids = set(s1[s1['country'].str.lower().str.strip() == 'france']['entity_id'])
        
        # Calculate how many of the 0.98 predicted matches belong to France
        france_preds = 0
        total_preds = 0
        for idx, row in res.iterrows():
            s1_id = row['source1_entity_id']
            matches = str(row['matched_entity_ids']).split(',')
            
            total_preds += len(matches)
            if s1_id in france_s1_ids:
                france_preds += len(matches)
                
        print(f"Percentage of current 0.98 predicted matches belonging to France: {france_preds / total_preds * 100:.2f}% (Total preds: {total_preds})")
    
    # Average candidates per French S1 (we can estimate by reading a chunk of candidate_pairs.tsv)
    cand_path = os.path.join(base_dir, "student_resource", "candidate_pairs.tsv")
    if os.path.exists(cand_path):
        print("\nEstimating average candidates for French S1 (first 1 million candidates)...")
        cand_chunk = pd.read_csv(cand_path, sep='\t', nrows=1000000)
        
        france_s1_ids = set(s1[s1['country'].str.lower().str.strip() == 'france']['entity_id'])
        cand_chunk['is_france'] = cand_chunk['source1_entity_id'].isin(france_s1_ids)
        
        france_cands = cand_chunk[cand_chunk['is_france']]
        unique_france_s1 = france_cands['source1_entity_id'].nunique()
        if unique_france_s1 > 0:
            avg_cands = len(france_cands) / unique_france_s1
            print(f"Avg candidates per French S1 (estimated from chunk): {avg_cands:.2f}")

if __name__ == "__main__":
    main()
