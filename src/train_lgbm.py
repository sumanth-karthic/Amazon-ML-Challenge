import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score, confusion_matrix

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "dataset", "train_features.parquet")
    
    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    
    # We want to identify the best features and exclude non-numeric/identifiers
    exclude_cols = ['s1_entity_id', 'cand_entity_id', 'label']
    features = [c for c in df.columns if c not in exclude_cols]
    
    total_rows = len(df)
    total_pos = int(df['label'].sum())
    total_neg = total_rows - total_pos
    pos_rate = total_pos / total_rows if total_rows > 0 else 0
    
    print(f"Total Rows:        {total_rows:,}")
    print(f"Total Features:    {len(features)}")
    print(f"Positive count:    {total_pos:,} ({pos_rate:.4%})")
    print(f"Negative count:    {total_neg:,}")
    
    # 6. Entity-level split
    print("\nSplitting train/val by s1_entity_id...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['s1_entity_id']))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    train_entities = df_train['s1_entity_id'].nunique()
    val_entities = df_val['s1_entity_id'].nunique()
    print(f"Train S1 Entities: {train_entities:,} ({len(df_train):,} pairs)")
    print(f"Val S1 Entities:   {val_entities:,} ({len(df_val):,} pairs)")
    
    # 5. Class imbalance scale_pos_weight
    train_pos = int(df_train['label'].sum())
    train_neg = len(df_train) - train_pos
    scale_pos_weight = train_neg / train_pos if train_pos > 0 else 1.0
    print(f"Train scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 7. Train Baseline LightGBM
    X_train, y_train = df_train[features], df_train['label']
    X_val, y_val = df_val[features], df_val['label']
    
    print("\nTraining LightGBM model...")
    t0 = time.time()
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # We use early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='aucpr',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    print(f"Training completed in {time.time() - t0:.2f} seconds.")
    
    print("\nPredicting on validation set...")
    # Predict probabilities
    y_val_probs = model.predict_proba(X_val)[:, 1]
    df_val = df_val.copy()
    df_val['pred_prob'] = y_val_probs
    
    # 8-10. Threshold tuning
    print("\nThreshold Tuning on Validation Set:")
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f05 = -1
    best_thresh = 0
    best_preds = None
    
    for t in thresholds:
        preds = (y_val_probs >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        
        print(f"Thresh={t:.2f} | P: {p:.4f} | R: {r:.4f} | F0.5: {f05:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            best_preds = preds
            
    print(f"\nBest Threshold: {best_thresh:.2f} (F0.5 = {best_f05:.4f})")
    
    # Evaluate at best threshold
    p = precision_score(y_val, best_preds, zero_division=0)
    r = recall_score(y_val, best_preds, zero_division=0)
    pr_auc = average_precision_score(y_val, y_val_probs)
    cm = confusion_matrix(y_val, best_preds)
    
    print("\nValidation Performance at Best Threshold:")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F0.5:      {best_f05:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 11. Sub-population reports
    print("\nSub-population Performance (Best Threshold):")
    
    def evaluate_subpop(mask, name):
        if mask.sum() == 0:
            print(f"{name}: No validation samples")
            return
        sub_y = y_val[mask]
        sub_p = best_preds[mask]
        sub_prec = precision_score(sub_y, sub_p, zero_division=0)
        sub_rec = recall_score(sub_y, sub_p, zero_division=0)
        sub_f05 = fbeta_score(sub_y, sub_p, beta=0.5, zero_division=0)
        print(f"[{name}] Pairs: {mask.sum():,} | Pos: {sub_y.sum():,} | P: {sub_prec:.4f} | R: {sub_rec:.4f} | F0.5: {sub_f05:.4f}")
        
    mask_s2 = df_val['cand_entity_id'].str.startswith('S2-')
    mask_s3 = df_val['cand_entity_id'].str.startswith('S3-')
    mask_addr_miss = df_val['address_missing'] == 1
    
    evaluate_subpop(mask_s2, "Source 2 Candidates")
    evaluate_subpop(mask_s3, "Source 3 Candidates")
    evaluate_subpop(mask_addr_miss, "Missing Address")
    
    # Singletons: S1 entities with 0 positive pairs in validation set
    s1_pos_counts = df_val.groupby('s1_entity_id')['label'].sum()
    singleton_s1s = s1_pos_counts[s1_pos_counts == 0].index
    mask_singleton = df_val['s1_entity_id'].isin(singleton_s1s)
    
    print(f"[Singleton S1 Entities (No positive candidate in pool)]")
    print(f"Total such S1 entities in val: {len(singleton_s1s):,} / {val_entities:,}")
    print(f"False Positives generated for singletons: {best_preds[mask_singleton].sum():,}")
    
    # 12. Save models and outputs
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "lgbm_model.txt")
    model.booster_.save_model(model_path)
    print(f"\nSaved model to: {model_path}")
    
    preds_path = os.path.join(models_dir, "val_predictions.parquet")
    df_val_out = df_val[['s1_entity_id', 'cand_entity_id', 'label', 'pred_prob']]
    df_val_out.to_parquet(preds_path, index=False)
    print(f"Saved validation predictions to: {preds_path}")
    
    # Feature Importances
    print("\nFeature Importances:")
    importances = model.feature_importances_
    feat_imps = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    for _, row in feat_imps.head(10).iterrows():
        print(f"  {row['feature']:<30}: {row['importance']}")

if __name__ == '__main__':
    train()
