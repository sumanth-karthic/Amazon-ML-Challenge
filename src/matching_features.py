import rapidfuzz

def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    union_len = len(set1.union(set2))
    if union_len == 0:
        return 0.0
    return len(set1.intersection(set2)) / union_len

def compute_idf_overlap(set1, set2, idf_dict):
    intersect = set1.intersection(set2)
    return sum(idf_dict.get(token, 0.5) for token in intersect)

def edit_similarity(s1, s2):
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 0.0
    # rapidfuzz is extremely fast and drops-in for difflib ratio
    return rapidfuzz.fuzz.ratio(s1, s2) / 100.0

class PairFeatureGenerator:
    def __init__(self, idf_dict):
        self.idf_dict = idf_dict
        
    def generate_features(self, s1_meta, s2_meta, blocking_score, blocking_rank):
        """
        Generates pair-level features for ML classification.
        s1_meta, s2_meta: dictionaries containing entity metadata 
                          (e.g., from iterrows or a DB fetch)
        """
        features = {}
        
        # Safely extract with fallbacks
        s1_name = str(s1_meta.get('name_norm', ''))
        s1_addr = str(s1_meta.get('address_norm', '')) # We'll assume preprocessing makes this available
        def get_ngrams(text, n=3):
            if not text: return []
            text = f" {text} "
            return [text[i:i+n] for i in range(len(text)-n+1)]

        s1_nt = set(s1_name.split())
        s1_at = set(s1_addr.split())
        s1_ng = set(get_ngrams(s1_name))
        s1_country = str(s1_meta.get('country', '')).lower()
        s1_postal = str(s1_meta.get('postal_code', ''))
        s1_addr_miss = s1_meta.get('address_missing', False)
        
        s2_name = str(s2_meta.get('name_norm', ''))
        s2_addr = str(s2_meta.get('address_norm', ''))
        if not s2_addr and 'business_address' in s2_meta:
            s2_addr = str(s2_meta.get('business_address', '')).lower()
            
        s2_nt = set(s2_name.split())
        s2_at = set(s2_addr.split())
        s2_ng = set(get_ngrams(s2_name))
        s2_country = str(s2_meta.get('country', '')).lower()
        s2_postal = str(s2_meta.get('postal_code', ''))
        s2_addr_miss = s2_meta.get('address_missing', False)
        
        # 1. Business-name similarity
        features['name_exact_match'] = 1 if s1_name == s2_name and s1_name else 0
        features['name_edit_sim'] = edit_similarity(s1_name, s2_name)
        features['name_token_jaccard'] = compute_jaccard(s1_nt, s2_nt)
        features['name_token_overlap_count'] = len(s1_nt.intersection(s2_nt))
        features['name_idf_overlap'] = compute_idf_overlap(s1_nt, s2_nt, self.idf_dict)
        features['name_ngram_jaccard'] = compute_jaccard(s1_ng, s2_ng)
        
        # 2. Address similarity
        features['addr_exact_match'] = 1 if s1_addr == s2_addr and s1_addr else 0
        features['addr_edit_sim'] = edit_similarity(s1_addr, s2_addr)
        features['addr_token_jaccard'] = compute_jaccard(s1_at, s2_at)
        features['addr_idf_overlap'] = compute_idf_overlap(s1_at, s2_at, self.idf_dict)
        
        features['postal_exact_match'] = 1 if s1_postal == s2_postal and s1_postal else 0
        features['postal_missing'] = 1 if not s1_postal or not s2_postal else 0
        features['address_missing'] = 1 if s1_addr_miss or s2_addr_miss else 0
        
        # 3. Country
        features['country_exact_match'] = 1 if s1_country == s2_country and s1_country else 0
        features['country_missing'] = 1 if not s1_country or not s2_country else 0
        
        # 4. Combined / Interactions
        features['name_exact_and_addr_exact'] = features['name_exact_match'] * features['addr_exact_match']
        features['name_sim_x_addr_sim'] = features['name_edit_sim'] * features['addr_edit_sim']
        features['strong_name_weak_addr'] = 1 if features['name_edit_sim'] > 0.85 and features['addr_edit_sim'] < 0.5 else 0
        features['strong_addr_weak_name'] = 1 if features['addr_edit_sim'] > 0.85 and features['name_edit_sim'] < 0.5 else 0
        
        # 5. Blocking features
        features['blocking_score'] = float(blocking_score)
        features['blocking_rank'] = int(blocking_rank)
        
        return features

    def generate_features_batch(self, s1_df, s2_df, blocking_scores, blocking_ranks):
        """
        Vectorized/batched generation of features to avoid Pandas loc/iterrows overhead.
        s1_df and s2_df are DataFrames aligned such that s1_df.iloc[i] is matched with s2_df.iloc[i].
        """
        import pandas as pd
        
        # Convert columns to flat lists for fast Python looping
        s1_name = s1_df['name_norm'].fillna('').astype(str).tolist()
        s1_addr = s1_df['address_norm'].fillna('').astype(str).tolist()
        s1_country = s1_df['country'].fillna('').astype(str).str.lower().tolist()
        s1_postal = s1_df['postal_code'].fillna('').astype(str).tolist()
        s1_addr_miss = s1_df['address_missing'].fillna(False).astype(bool).tolist()
        
        s2_name = s2_df['name_norm'].fillna('').astype(str).tolist()
        
        if 'business_address' in s2_df.columns:
            s2_addr_raw = s2_df['address_norm'].fillna('')
            s2_addr_fallback = s2_df['business_address'].fillna('').astype(str).str.lower()
            s2_addr = s2_addr_raw.where(s2_addr_raw != '', s2_addr_fallback).astype(str).tolist()
        else:
            s2_addr = s2_df['address_norm'].fillna('').astype(str).tolist()
            
        s2_country = s2_df['country'].fillna('').astype(str).str.lower().tolist()
        s2_postal = s2_df['postal_code'].fillna('').astype(str).tolist()
        s2_addr_miss = s2_df['address_missing'].fillna(False).astype(bool).tolist()
        
        def get_ngrams(text, n=3):
            if not text: return []
            text = f" {text} "
            return [text[i:i+n] for i in range(len(text)-n+1)]

        # Precompute sets
        s1_nt = [set(s.split()) for s in s1_name]
        s1_at = [set(s.split()) for s in s1_addr]
        s1_ng = [set(get_ngrams(s)) for s in s1_name]
        
        s2_nt = [set(s.split()) for s in s2_name]
        s2_at = [set(s.split()) for s in s2_addr]
        s2_ng = [set(get_ngrams(s)) for s in s2_name]
        
        # Feature computation via list comprehensions
        name_exact_match = [1 if n1 == n2 and n1 else 0 for n1, n2 in zip(s1_name, s2_name)]
        name_edit_sim = [edit_similarity(n1, n2) for n1, n2 in zip(s1_name, s2_name)]
        name_token_jaccard = [compute_jaccard(st1, st2) for st1, st2 in zip(s1_nt, s2_nt)]
        name_token_overlap_count = [len(st1.intersection(st2)) for st1, st2 in zip(s1_nt, s2_nt)]
        name_idf_overlap = [compute_idf_overlap(st1, st2, self.idf_dict) for st1, st2 in zip(s1_nt, s2_nt)]
        name_ngram_jaccard = [compute_jaccard(sg1, sg2) for sg1, sg2 in zip(s1_ng, s2_ng)]
        
        addr_exact_match = [1 if a1 == a2 and a1 else 0 for a1, a2 in zip(s1_addr, s2_addr)]
        addr_edit_sim = [edit_similarity(a1, a2) for a1, a2 in zip(s1_addr, s2_addr)]
        addr_token_jaccard = [compute_jaccard(at1, at2) for at1, at2 in zip(s1_at, s2_at)]
        addr_idf_overlap = [compute_idf_overlap(at1, at2, self.idf_dict) for at1, at2 in zip(s1_at, s2_at)]
        
        postal_exact_match = [1 if p1 == p2 and p1 else 0 for p1, p2 in zip(s1_postal, s2_postal)]
        postal_missing = [1 if not p1 or not p2 else 0 for p1, p2 in zip(s1_postal, s2_postal)]
        address_missing_feat = [1 if m1 or m2 else 0 for m1, m2 in zip(s1_addr_miss, s2_addr_miss)]
        
        country_exact_match = [1 if c1 == c2 and c1 else 0 for c1, c2 in zip(s1_country, s2_country)]
        country_missing = [1 if not c1 or not c2 else 0 for c1, c2 in zip(s1_country, s2_country)]
        
        name_exact_and_addr_exact = [n * a for n, a in zip(name_exact_match, addr_exact_match)]
        name_sim_x_addr_sim = [n * a for n, a in zip(name_edit_sim, addr_edit_sim)]
        strong_name_weak_addr = [1 if n > 0.85 and a < 0.5 else 0 for n, a in zip(name_edit_sim, addr_edit_sim)]
        strong_addr_weak_name = [1 if a > 0.85 and n < 0.5 else 0 for n, a in zip(name_edit_sim, addr_edit_sim)]
        
        return pd.DataFrame({
            'name_exact_match': name_exact_match,
            'name_edit_sim': name_edit_sim,
            'name_token_jaccard': name_token_jaccard,
            'name_token_overlap_count': name_token_overlap_count,
            'name_idf_overlap': name_idf_overlap,
            'name_ngram_jaccard': name_ngram_jaccard,
            
            'addr_exact_match': addr_exact_match,
            'addr_edit_sim': addr_edit_sim,
            'addr_token_jaccard': addr_token_jaccard,
            'addr_idf_overlap': addr_idf_overlap,
            
            'postal_exact_match': postal_exact_match,
            'postal_missing': postal_missing,
            'address_missing': address_missing_feat,
            
            'country_exact_match': country_exact_match,
            'country_missing': country_missing,
            
            'name_exact_and_addr_exact': name_exact_and_addr_exact,
            'name_sim_x_addr_sim': name_sim_x_addr_sim,
            'strong_name_weak_addr': strong_name_weak_addr,
            'strong_addr_weak_name': strong_addr_weak_name,
            
            'blocking_score': blocking_scores,
            'blocking_rank': blocking_ranks
        })
