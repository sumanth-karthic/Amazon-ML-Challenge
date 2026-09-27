import pandas as pd
import numpy as np

print("--- 5-7. Test Predictions Analysis ---")
s1_test = pd.read_csv('../dataset/test/test_source1.tsv', sep='\t', dtype=str).fillna('')
s1_test = s1_test[['entity_id', 'country']].rename(columns={'entity_id': 'source1_entity_id'})

# Load the three threshold files
res_098 = pd.read_csv('../output/threshold_tests/matching_098.tsv', sep='\t', names=['source1_entity_id', 'matched_entity_ids'], dtype=str)
# Note: matching_098.tsv might not have all 1.7M rows, we need to merge it with s1_test
res_098 = s1_test[['source1_entity_id']].merge(res_098, on='source1_entity_id', how='left').fillna('')

res_099 = pd.read_csv('../output/threshold_tests/matching_results_0.99.tsv', sep='\t', dtype=str).fillna('')
res_0995 = pd.read_csv('../output/threshold_tests/matching_results_0.995.tsv', sep='\t', dtype=str).fillna('')

def analyze_matches(df, th_name):
    # count matches
    df['match_count'] = df['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if str(x).strip() != '' else 0)
    df_merged = df.merge(s1_test, on='source1_entity_id')
    country_counts = df_merged[df_merged['match_count'] > 0].groupby('country').size()
    avg_matches = df['match_count'].mean()
    max_matches = df['match_count'].max()
    zero_matches = (df['match_count'] == 0).sum()
    print(f"\n[{th_name}]")
    print(f"Average Matches per S1: {avg_matches:.4f}")
    print(f"Max Matches for single S1: {max_matches}")
    print(f"Zero Match S1s: {zero_matches} ({(zero_matches/len(df))*100:.2f}%)")
    print("Match counts by Country:")
    for c, count in country_counts.items():
        print(f"  {c}: {count}")
    return df['matched_entity_ids'].values

m98 = analyze_matches(res_098, "Threshold 0.98")
m99 = analyze_matches(res_099, "Threshold 0.99")
m995 = analyze_matches(res_0995, "Threshold 0.995")

changed_98_99 = np.sum(m98 != m99)
changed_99_995 = np.sum(m99 != m995)
print(f"\nEntities changed from 0.98 to 0.99: {changed_98_99}")
print(f"Entities changed from 0.99 to 0.995: {changed_99_995}")
