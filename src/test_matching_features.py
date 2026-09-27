import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from matching_features import PairFeatureGenerator

def test_feature_generator():
    print("Running explicit test cases for PairFeatureGenerator...")
    # Dummy IDF dictionary for testing
    idf_dict = {'tech': 3.0, 'solutions': 2.0, 'global': 1.5, 'rue': 1.0, 'paris': 2.5}
    generator = PairFeatureGenerator(idf_dict)
    
    # 1. Exact Match Test
    s1_exact = {
        'name_norm': 'tech solutions global', 'name_tokens': ['tech', 'solutions', 'global'],
        'address_norm': '10 rue de paris', 'address_tokens': ['10', 'rue', 'de', 'paris'],
        'ngrams': ['tec', 'ech', 'sol', 'olu', 'lut'],
        'country': 'france', 'postal_code': '75001', 'address_missing': False
    }
    f_exact = generator.generate_features(s1_exact, s1_exact, blocking_score=15.0, blocking_rank=1)
    assert f_exact['name_exact_match'] == 1
    assert f_exact['addr_exact_match'] == 1
    assert f_exact['country_exact_match'] == 1
    assert f_exact['name_token_jaccard'] == 1.0
    print("[OK] Exact match test passed")
    
    # 2. Typo Match Test
    s2_typo = {
        'name_norm': 'tech solutons glbal', 'name_tokens': ['tech', 'solutons', 'glbal'],
        'address_norm': '10 rue de pparis', 'address_tokens': ['10', 'rue', 'de', 'pparis'],
        'ngrams': ['tec', 'ech', 'sol', 'olu', 'glb'],
        'country': 'france', 'postal_code': '75001', 'address_missing': False
    }
    f_typo = generator.generate_features(s1_exact, s2_typo, blocking_score=10.0, blocking_rank=2)
    assert f_typo['name_exact_match'] == 0
    assert f_typo['name_edit_sim'] > 0.8
    assert f_typo['name_token_jaccard'] < 1.0
    print("[OK] Typo match test passed")
    
    # 3. Missing Address Test
    s1_miss = {'name_norm': 'hello', 'address_missing': True}
    s2_miss = {'name_norm': 'hello', 'address_missing': False}
    f_miss = generator.generate_features(s1_miss, s2_miss, blocking_score=5.0, blocking_rank=3)
    assert f_miss['address_missing'] == 1
    print("[OK] Missing address test passed")
    
    # 4. Different Countries Test
    s1_ctry = {'name_norm': 'corp', 'country': 'us'}
    s2_ctry = {'name_norm': 'corp', 'country': 'india'}
    f_ctry = generator.generate_features(s1_ctry, s2_ctry, blocking_score=5.0, blocking_rank=4)
    assert f_ctry['country_exact_match'] == 0
    assert f_ctry['country_missing'] == 0
    print("[OK] Different countries test passed")
    
    # 5. Multilingual Names Test
    s1_multi = {'name_norm': 'लाल किला', 'name_tokens': ['लाल', 'किला']}
    s2_multi = {'name_norm': 'लाल किला', 'name_tokens': ['लाल', 'किला']}
    f_multi = generator.generate_features(s1_multi, s2_multi, blocking_score=10.0, blocking_rank=1)
    assert f_multi['name_exact_match'] == 1
    assert f_multi['name_edit_sim'] == 1.0
    print("[OK] Multilingual names test passed\n")
    
def generate_sample_dataset():
    from evaluate_blocking_v2_2 import load_benchmark_data, print_memory
    from preprocessing import preprocess_chunk as preprocess_pipeline
    import time
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Loading datasets for feature generation...")
    t0 = time.time()
    
    # For the feature sample, we don't need to generate 2.5M pairs.
    # We will just process the first 1,000 S1 queries against a small S2/S3 index
    # to demonstrate the chunking and feature output.
    
    # Instead of running the full evaluation script, we will just use 
    # the existing blocking model to get some pairs, or we can just 
    # fetch the top 10 rows and make some Cartesian products.
    # But to do it properly as a candidate pair generator:
    
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    
    s1_df = pd.read_csv(s1_path, sep='\t').head(1000)
    s1_df['address_missing'] = s1_df['business_address'].isna() | (s1_df['business_address'] == "")
    s1_df = preprocess_pipeline(s1_df)
    
    s2_df = pd.read_csv(s2_path, sep='\t').head(10000)
    s2_df['address_missing'] = s2_df['business_address'].isna() | (s2_df['business_address'] == "")
    s2_df = preprocess_pipeline(s2_df)
    
    print(f"Loaded 1k S1 and 10k S2 records in {time.time()-t0:.2f}s")
    
    # Build dictionary representation of S2 to simulate O(1) lookup
    # In full production, this would be SQLite or partitioned Parquet
    s2_store = {}
    for row in s2_df.to_dict(orient='records'):
        s2_store[row['entity_id']] = row
        
    print("Generating sample features in chunks...")
    t_feat = time.time()
    
    idf_dict = {} # Mock IDF dict
    generator = PairFeatureGenerator(idf_dict)
    
    feature_rows = []
    
    # Generate 50 random pairs per S1 row to simulate candidates
    np.random.seed(42)
    s2_keys = list(s2_store.keys())
    
    for s1_idx, s1_row in enumerate(s1_df.to_dict(orient='records')):
        # Mock candidate retrieval
        cands = np.random.choice(s2_keys, size=5, replace=False)
        for rank, cand_id in enumerate(cands, 1):
            s2_row = s2_store[cand_id]
            feat = generator.generate_features(s1_row, s2_row, blocking_score=10.0/rank, blocking_rank=rank)
            
            # Ground truth label logic would happen here in production:
            # label = 1 if cand_id in train_ground_truth[s1_row['entity_id']] else 0
            # feat['label'] = label
            
            feat['s1_entity_id'] = s1_row['entity_id']
            feat['cand_entity_id'] = cand_id
            feature_rows.append(feat)
            
    feat_df = pd.DataFrame(feature_rows)
    print(f"Feature generation for 5k pairs completed in {time.time()-t_feat:.2f}s")
    print(f"Peak RAM during feature gen: {print_memory():.1f} MB")
    
    # Put IDs at the front
    cols = ['s1_entity_id', 'cand_entity_id'] + [c for c in feat_df.columns if c not in ['s1_entity_id', 'cand_entity_id']]
    feat_df = feat_df[cols]
    
    print("\n" + "="*80)
    print("SAMPLE FEATURE OUTPUT (First 3 rows)")
    print("="*80)
    print(feat_df.head(3).T)
    
    print("\n" + "="*80)
    print("SCHEMA & DATATYPES")
    print("="*80)
    print(feat_df.dtypes)
    
    print("\n" + "="*80)
    print("MISSING VALUES")
    print("="*80)
    print(feat_df.isna().sum())
    
    output_path = os.path.join(base_dir, 'reports', 'matching_features_sample.csv')
    feat_df.to_csv(output_path, index=False)
    print(f"\nSample saved to: {output_path}")

if __name__ == "__main__":
    test_feature_generator()
    generate_sample_dataset()
