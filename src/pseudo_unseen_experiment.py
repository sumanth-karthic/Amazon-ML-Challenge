import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score

def compute_metrics(probs, labels, thresholds, max_recall_denominator=None):
    results = {}
    for th in thresholds:
        preds = (probs >= th).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        if max_recall_denominator is not None:
            rec = tp / max_recall_denominator
        else:
            fn = np.sum((preds == 0) & (labels == 1))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        results[th] = (prec, rec, f05)
    return results

def run_experiment(name, train_df, test_df, features, test_gt_count):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Train LightGBM
    X_train = train_df[features]
    y_train = train_df['label']
    
    X_test = test_df[features]
    y_test = test_df['label']
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'learning_rate': 0.1,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 200,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # 3. Calculate metrics
    thresholds = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    metrics = compute_metrics(probs, y_test, thresholds, max_recall_denominator=test_gt_count)
    
    print("\nMetrics on Test Set (Pseudo-Unseen Country):")
    print("Thresh | Precision | Recall  | F0.5")
    print("-" * 45)
    for th in thresholds:
        p, r, f = metrics[th]
        print(f"{th:.3f}  | {p:.4f}    | {r:.4f}  | {f:.4f}")
        
    # 4. Probability distribution
    print("\nProbability Distribution (Positive Pairs):")
    pos_probs = probs[y_test == 1]
    neg_probs = probs[y_test == 0]
    
    for th in [0.5, 0.9, 0.95, 0.98, 0.99]:
        pos_above = np.mean(pos_probs >= th)
        print(f"  >= {th:.2f}: {pos_above*100:.1f}%")
        
    print("\nProbability Distribution (Negative Pairs):")
    for th in [0.5, 0.9, 0.95, 0.98, 0.99]:
        neg_above = np.mean(neg_probs >= th)
        print(f"  >= {th:.2f}: {neg_above*100:.3f}%")
        
    return model, probs, y_test

def main():
    base_dir = "../"
    
    print("Loading datasets...")
    # Load S1 to get country
    s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'country', 'business_name', 'business_address'])
    
    # We also need exact normalized name for the rule test, let's just use raw lowercase for simplicity
    s1_df['name_norm'] = s1_df['business_name'].fillna('').str.lower()
    s1_df['addr_norm'] = s1_df['business_address'].fillna('').str.lower()
    
    s1_dict = s1_df.set_index('entity_id').to_dict('index')
    
    gt_path = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep='\t')
    
    # Count GT pairs by country
    gt_count_us = 0
    gt_count_india = 0
    for idx, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if pd.isna(matches_str) or not str(matches_str).strip():
            continue
        targets = [x.strip() for x in str(matches_str).split(',')]
        if s1 in s1_dict:
            c = str(s1_dict[s1]['country']).lower().strip()
            if c == 'us':
                gt_count_us += len(targets)
            elif c == 'india':
                gt_count_india += len(targets)
                
    features_path = os.path.join(base_dir, "dataset", "train_features.parquet")
    df = pd.read_parquet(features_path)
    
    df['s1_country'] = df['s1_entity_id'].map(lambda x: str(s1_dict.get(x, {}).get('country', '')).lower().strip())
    
    us_df = df[df['s1_country'] == 'us']
    in_df = df[df['s1_country'] == 'india']
    
    features = [c for c in df.columns if c not in ['s1_entity_id', 'cand_entity_id', 'label', 's1_country']]
    
    # Experiment A: Train US, Eval India
    model_a, probs_a, y_a = run_experiment("A: Train US -> Eval India", us_df, in_df, features, gt_count_india)
    
    # Experiment B: Train India, Eval US
    model_b, probs_b, y_b = run_experiment("B: Train India -> Eval US", in_df, us_df, features, gt_count_us)
    
    # 5. Evaluate Decision Rules on India (Exp A results)
    print("\n" + "="*60)
    print("TESTING DECISION RULES ON pseudo-unseen (INDIA)")
    print("="*60)
    
    # We will test on in_df
    # Add predicted probabilities back to in_df
    in_df = in_df.copy()
    in_df['prob'] = probs_a
    
    rules = [
        ("A. global 0.98", lambda row: row['prob'] >= 0.98),
        ("B. global 0.99", lambda row: row['prob'] >= 0.99),
        ("C. global 0.995", lambda row: row['prob'] >= 0.995),
        ("D. high threshold + exact name (0.995 OR exact_name)", lambda row: row['prob'] >= 0.995 or row['name_exact_match'] == 1),
        ("E. high threshold + name/addr similarity (0.995 OR (jw_name>0.9 & jw_addr>0.9))", 
         lambda row: row['prob'] >= 0.995 or (row.get('name_jaro_winkler', 0) > 0.9 and row.get('address_jaro_winkler', 0) > 0.9))
    ]
    
    for name, rule_func in rules:
        preds = in_df.apply(rule_func, axis=1).astype(int)
        tp = np.sum((preds == 1) & (in_df['label'] == 1))
        fp = np.sum((preds == 1) & (in_df['label'] == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / gt_count_india
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:65s}: Prec={prec:.4f}, Rec={rec:.4f}, F0.5={f05:.4f}")

if __name__ == "__main__":
    main()
