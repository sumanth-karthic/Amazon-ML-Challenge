import os
import collections

os.makedirs('../output', exist_ok=True)

# 1. Fix matching_results.tsv
print("Fixing matching_results.tsv...")
total_matching_rows = 0
with open('../matching_results.tsv', 'r', encoding='utf-8') as f_in, \
     open('../output/matching_results.tsv', 'w', encoding='utf-8') as f_out:
    
    header = f_in.readline()
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        line = line.strip('\n')
        if not line: continue
        parts = line.split('\t')
        s1 = parts[0]
        matches = parts[1] if len(parts) > 1 else ""
        
        if matches:
            m_list = matches.split()
            m_uniq = list(dict.fromkeys(m_list))
            matches_comma = ",".join(m_uniq)
        else:
            matches_comma = ""
            
        f_out.write(f"{s1}\t{matches_comma}\n")
        total_matching_rows += 1

# 2. Rebuild candidate_pairs.tsv
print("Rebuilding candidate_pairs.tsv...")
s1_order = []
with open('../dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_s1:
    f_s1.readline()
    for line in f_s1:
        s1_order.append(line.split('\t')[0])

s1_iter = iter(s1_order)
expected_s1 = next(s1_iter, None)

current_s1 = None
current_cands = []
total_cand_rows = 0
total_cand_ids = 0
empty_cands = 0

def write_and_advance(s1, cands, f_out):
    global total_cand_rows, total_cand_ids, empty_cands
    c_uniq = list(dict.fromkeys(cands))
    f_out.write(f"{s1}\t{','.join(c_uniq)}\n")
    total_cand_rows += 1
    total_cand_ids += len(c_uniq)
    if not c_uniq:
        empty_cands += 1

with open('../candidate_pairs.tsv', 'r', encoding='utf-8') as f_in, \
     open('../output/candidate_pairs.tsv', 'w', encoding='utf-8') as f_out:
    
    f_in.readline()
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    
    for line in f_in:
        line = line.strip('\n')
        if not line: continue
        parts = line.split('\t')
        s1 = parts[0]
        cand = parts[1]
        
        if current_s1 != s1:
            if current_s1 is not None:
                while expected_s1 and expected_s1 != current_s1:
                    write_and_advance(expected_s1, [], f_out)
                    expected_s1 = next(s1_iter, None)
                
                write_and_advance(current_s1, current_cands, f_out)
                expected_s1 = next(s1_iter, None)
            
            current_s1 = s1
            current_cands = [cand]
        else:
            current_cands.append(cand)
            
    if current_s1 is not None:
        while expected_s1 and expected_s1 != current_s1:
            write_and_advance(expected_s1, [], f_out)
            expected_s1 = next(s1_iter, None)
            
        write_and_advance(current_s1, current_cands, f_out)
        expected_s1 = next(s1_iter, None)
        
    while expected_s1 is not None:
        write_and_advance(expected_s1, [], f_out)
        expected_s1 = next(s1_iter, None)

print(f"final matching_results row count: {total_matching_rows}")
print(f"final candidate_pairs row count: {total_cand_rows}")
print(f"number of empty candidate lists: {empty_cands}")
print(f"total candidate IDs represented: {total_cand_ids}")

# Verify consistency
consistent = True
with open('../output/matching_results.tsv', 'r') as fm, open('../output/candidate_pairs.tsv', 'r') as fc:
    fm.readline()
    fc.readline()
    for lm, lc in zip(fm, fc):
        sm, mm = lm.strip('\n').split('\t')
        sc, mc = lc.strip('\n').split('\t')
        if sm != sc:
            print("ERROR: S1 mismatch in parallel verification")
            break
        mm_list = mm.split(',') if mm else []
        mc_list = set(mc.split(',') if mc else [])
        for m in mm_list:
            if m not in mc_list:
                consistent = False
                break
        if not consistent:
            break
            
print(f"whether every predicted match is contained in the candidate list: {consistent}")

s_mr = os.path.getsize('../output/matching_results.tsv')
s_cp = os.path.getsize('../output/candidate_pairs.tsv')
print(f"final file sizes: matching_results.tsv={s_mr/1024/1024:.2f}MB, candidate_pairs.tsv={s_cp/1024/1024:.2f}MB")
