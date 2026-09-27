import sys
import os
import time
import pandas as pd
import json

# Add current directory to path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import normalize_text, extract_postal_code, preprocess_chunk, preprocess_file

def test_normalization():
    # punctuation normalization
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("A.B.C.") == "a b c"
    assert normalize_text("a&b") == "a and b"
    assert normalize_text("test_underscore") == "test underscore"
    
    # whitespace normalization
    assert normalize_text("  spaces   here  ") == "spaces here"
    assert normalize_text("tab\tseparated") == "tab separated"
    
    # Unicode preservation
    # Devanagari text
    devanagari = "नमस्ते दुनिया"
    assert normalize_text(devanagari) == devanagari
    
    # missing values
    assert normalize_text(None) == ""
    assert normalize_text(float('nan')) == ""
    
    # numeric/postal preservation
    assert normalize_text("PO Box 12345") == "po box 12345"
    assert normalize_text("12345-6789") == "12345 6789"
    
    # deterministic output
    assert normalize_text("A, B, C") == normalize_text("a, b, c")

def test_postal_extraction():
    assert extract_postal_code("New York, NY 10001") == "10001"
    assert extract_postal_code("Something 12345-6789") == "12345-6789"
    assert extract_postal_code("Delhi 110001") == "110001"
    assert extract_postal_code("Paris 75001") == "75001"
    assert extract_postal_code("Missing postal") == ""
    assert extract_postal_code(None) == ""
    assert extract_postal_code(float('nan')) == ""
    # Check it takes the last one if there are multiple
    assert extract_postal_code("12345 and 98765") == "98765"

def run_tests():
    print("Running unit tests...")
    test_normalization()
    test_postal_extraction()
    print("All unit tests passed!")

def run_performance_test(nrows):
    print(f"\n--- Running performance test for {nrows} rows ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = [
        os.path.join(base_dir, "dataset", "train", "train_source1.tsv"),
        os.path.join(base_dir, "dataset", "train", "train_source2.tsv"),
        os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    ]
    
    total_time = 0
    df_list = []
    
    for input_file in test_files:
        if not os.path.exists(input_file):
            print(f"Skipping {input_file} (does not exist)")
            continue
            
        start_time = time.time()
        
        # We'll just read into memory for the performance test to measure raw compute time
        df = pd.read_csv(input_file, sep='\t', nrows=nrows, dtype=str, quoting=3, on_bad_lines='warn')
        
        import tracemalloc
        tracemalloc.start()
        
        processed_df = preprocess_chunk(df)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"File: {os.path.basename(input_file)}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Peak memory usage (tracemalloc): {peak / 10**6:.2f} MB")
        
        df_list.append(processed_df)
        
    return df_list

def generate_report(dfs, num_rows):
    if not dfs:
        return
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base_dir, "reports", f"preprocessing_examples_{num_rows}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Preprocessing Examples\n")
        f.write("="*80 + "\n\n")
        
        for idx, df in enumerate(dfs):
            f.write(f"--- Source {idx+1} Examples ---\n")
            # Take 10 examples from each df
            sample = df.dropna(subset=['business_address']).sample(10, random_state=42)
            for i, row in sample.iterrows():
                f.write(f"Original Name:    {row['business_name']}\n")
                f.write(f"Normalized Name:  {row['name_norm']}\n")
                f.write(f"Tokens:           {row['name_tokens']}\n")
                f.write(f"Original Addr:    {row['business_address']}\n")
                f.write(f"Normalized Addr:  {row['address_norm']}\n")
                f.write(f"Tokens:           {row['address_tokens']}\n")
                f.write(f"Postal Code:      {row['postal_code']}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")
            
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    run_tests()
    
    dfs_10k = run_performance_test(10000)
    generate_report(dfs_10k, 10000)
    
    run_performance_test(100000)
