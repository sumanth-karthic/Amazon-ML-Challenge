import os
import pandas as pd
import json

base_dir = r"c:/Users/suman/OneDrive/Desktop/6ab10eb3b23ba_student_resource/student_resource"
report_dir = os.path.join(base_dir, "reports")
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "dataset_inspection.txt")

tsv_files = {
    "train_source1.tsv": os.path.join(base_dir, "dataset", "train", "train_source1.tsv"),
    "train_source2.tsv": os.path.join(base_dir, "dataset", "train", "train_source2.tsv"),
    "train_source3.tsv": os.path.join(base_dir, "dataset", "train", "train_source3.tsv"),
    "test_source1.tsv": os.path.join(base_dir, "dataset", "test", "test_source1.tsv"),
    "test_source2.tsv": os.path.join(base_dir, "dataset", "test", "test_source2.tsv"),
    "test_source3.tsv": os.path.join(base_dir, "dataset", "test", "test_source3.tsv"),
}

def inspect_tsv(file_name, file_path, f):
    f.write(f"--- Inspection for {file_name} ---\n")
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='warn')
    except Exception as e:
        f.write(f"Error loading {file_name}: {e}\n\n")
        return
        
    f.write(f"Number of rows: {len(df)}\n")
    f.write(f"Number of columns: {len(df.columns)}\n")
    f.write(f"Column names: {list(df.columns)}\n")
    
    f.write("Data types:\n")
    for col, dtype in df.dtypes.items():
        f.write(f"  {col}: {dtype}\n")
        
    f.write("Missing values:\n")
    for col, missing in df.isnull().sum().items():
        f.write(f"  {col}: {missing}\n")
        
    f.write("Duplicate entity IDs:\n")
    id_col = 'entity_id' if 'entity_id' in df.columns else 'id'
    if id_col in df.columns:
        dups = df.duplicated(subset=[id_col]).sum()
        f.write(f"  {dups}\n")
    else:
        f.write("  No 'entity_id' column found.\n")
        
    f.write("5 sample rows:\n")
    f.write(df.head(5).to_string() + "\n")
    
    if 'country' in df.columns:
        f.write(f"Unique country values: {df['country'].dropna().unique().tolist()}\n")
    else:
        f.write("No 'country' column found.\n")
        
    f.write("Basic statistics for string fields:\n")
    # For newer pandas versions, string types are under 'object' or 'string'
    for col in df.select_dtypes(include=['object', 'string']).columns:
        valid_data = df[col].dropna().astype(str)
        if len(valid_data) > 0:
            lengths = valid_data.str.len()
            f.write(f"  {col}:\n")
            f.write(f"    Unique counts: {valid_data.nunique()}\n")
            f.write(f"    Min length: {lengths.min()}\n")
            f.write(f"    Max length: {lengths.max()}\n")
            f.write(f"    Mean length: {lengths.mean():.2f}\n")
    f.write("\n")

def inspect_ground_truth(f):
    file_name = "train_ground_truth.tsv"
    file_path = os.path.join(base_dir, "dataset", "train", file_name)
    f.write(f"--- Inspection for {file_name} ---\n")
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='warn')
    except Exception as e:
        f.write(f"Error loading {file_name}: {e}\n\n")
        return

    f.write(f"Number of rows: {len(df)}\n")
    f.write(f"Number of columns: {len(df.columns)}\n")
    f.write(f"Column names: {list(df.columns)}\n")
    
    f.write("Data types:\n")
    for col, dtype in df.dtypes.items():
        f.write(f"  {col}: {dtype}\n")
        
    f.write("Missing values:\n")
    for col, missing in df.isnull().sum().items():
        f.write(f"  {col}: {missing}\n")
        
    f.write("Duplicate entity IDs:\n")
    if 'source1_entity_id' in df.columns:
        dups = df.duplicated(subset=['source1_entity_id']).sum()
        f.write(f"  {dups}\n")
    else:
        f.write("  No 'source1_entity_id' column found.\n")

    f.write("5 sample rows:\n")
    f.write(df.head(5).to_string() + "\n")

    f.write("Analysis:\n")
    if 'source1_entity_id' in df.columns and 'matched_entity_ids' in df.columns:
        s1_ids = df['source1_entity_id']
        num_s1 = s1_ids.nunique()
        f.write(f"Number of Source 1 entities: {num_s1}\n")
        
        matches = df['matched_entity_ids'].fillna('')
        
        def count_matches(x):
            return len(str(x).split(',')) if str(x).strip() else 0
        
        match_counts = matches.apply(count_matches)
        
        singletons = (match_counts == 0).sum()
        f.write(f"Count singleton Source 1 entities: {singletons}\n")
        
        matches_s2 = matches.astype(str).str.contains('S2-').sum()
        matches_s3 = matches.astype(str).str.contains('S3-').sum()
        matches_both = (matches.astype(str).str.contains('S2-') & matches.astype(str).str.contains('S3-')).sum()
        
        f.write(f"Count entities matching Source 2: {matches_s2}\n")
        f.write(f"Count entities matching Source 3: {matches_s3}\n")
        f.write(f"Count entities matching both sources: {matches_both}\n")
        
        f.write(f"Distribution of number of matches:\n")
        dist = match_counts.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} matches: {v} entities\n")
    f.write("\n")

with open(report_path, "w", encoding='utf-8') as f:
    f.write("Project folder structure:\n")
    for root, dirs, files in os.walk(base_dir):
        # ignore macosx and hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        f.write(f"{indent}{os.path.basename(root)}/\n")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            if not file.startswith('._') and file != '.DS_Store' and file != '__MACOSX':
                f.write(f"{subindent}{file}\n")
    f.write("\n")
    
    inspect_ground_truth(f)
    for name, path in tsv_files.items():
        inspect_tsv(name, path, f)
