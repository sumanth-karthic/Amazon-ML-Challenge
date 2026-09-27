import collections
import math
import ast
import pandas as pd

def get_3grams(text):
    text = f"^{text}$"
    return [text[i:i+3] for i in range(len(text)-2)]

class BlockingStrategiesV2_1:
    def __init__(self, max_postings=5000, num_ngrams=5):
        self.max_postings = max_postings
        self.num_ngrams = num_ngrams
        
        # Frequencies
        self.name_token_df = collections.Counter()
        self.address_token_df = collections.Counter()
        self.name_ngram_df = collections.Counter()
        self.total_docs = 0
        
        # Indexes
        self.exact_name_idx = collections.defaultdict(list)
        
        # Country-aware and global token indexes
        self.global_token_idx = collections.defaultdict(list)
        self.country_token_idx = collections.defaultdict(list)
        
        self.ngram_idx = collections.defaultdict(list)
        self.postal_token_idx = collections.defaultdict(list)
        self.name_addr_idx = collections.defaultdict(list)
        
        # Store for quick scoring
        self.store = {}
        
    def get_idf(self, df_counter, token):
        df = df_counter.get(token, 0)
        return math.log((self.total_docs + 1) / (df + 1))
        
    def fit_frequencies(self, df):
        # using itertuples for speed
        for row in df.itertuples(index=False):
            # name_tokens
            nt_raw = getattr(row, 'name_tokens', [])
            if isinstance(nt_raw, str):
                nt_raw = ast.literal_eval(nt_raw) if nt_raw.startswith('[') else []
            name_tokens = set(nt_raw)
            for nt in name_tokens:
                if len(nt) > 1:
                    self.name_token_df[nt] += 1
                
            name_norm = getattr(row, 'name_norm', '')
            if pd.isna(name_norm): name_norm = ''
            
            if name_norm:
                ngrams = set(get_3grams(name_norm))
                for ng in ngrams:
                    self.name_ngram_df[ng] += 1
                    
            at_raw = getattr(row, 'address_tokens', [])
            if isinstance(at_raw, str):
                at_raw = ast.literal_eval(at_raw) if at_raw.startswith('[') else []
            addr_tokens = set(at_raw)
            for at in addr_tokens:
                if len(at) > 1:
                    self.address_token_df[at] += 1
                    
            self.total_docs += 1

    def build_indexes(self, df):
        for row in df.itertuples(index=False):
            eid = getattr(row, 'entity_id')
            
            name_norm = getattr(row, 'name_norm', '')
            if pd.isna(name_norm): name_norm = ''
            
            country = getattr(row, 'country', '')
            if pd.isna(country): country = ''
            country = country.lower()
            
            postal = getattr(row, 'postal_code', '')
            if pd.isna(postal): postal = ''
            
            nt_raw = getattr(row, 'name_tokens', [])
            if isinstance(nt_raw, str):
                nt_raw = ast.literal_eval(nt_raw) if nt_raw.startswith('[') else []
            name_tokens = [t for t in nt_raw if len(t) > 1]
            
            at_raw = getattr(row, 'address_tokens', [])
            if isinstance(at_raw, str):
                at_raw = ast.literal_eval(at_raw) if at_raw.startswith('[') else []
            addr_tokens = [t for t in at_raw if len(t) > 1]
            
            name_weight = sum(self.get_idf(self.name_token_df, t) for t in name_tokens)
            addr_weight = sum(self.get_idf(self.address_token_df, t) for t in addr_tokens)
            ngrams = set(get_3grams(name_norm)) if name_norm else set()
            
            self.store[eid] = {
                'name_tokens': set(name_tokens),
                'addr_tokens': set(addr_tokens),
                'country': country,
                'postal': postal,
                'name_norm': name_norm,
                'name_weight': name_weight,
                'addr_weight': addr_weight,
                'ngrams': ngrams
            }
            
            if name_norm:
                self.exact_name_idx[name_norm].append(eid)
                
            unique_name_tokens = list(set(name_tokens))
            if unique_name_tokens:
                unique_name_tokens.sort(key=lambda t: self.get_idf(self.name_token_df, t), reverse=True)
                for nt in unique_name_tokens[:3]: # top 3
                    self.global_token_idx[nt].append(eid)
                    if country:
                        self.country_token_idx[f"{country}|{nt}"].append(eid)
                    if postal:
                        self.postal_token_idx[f"{postal}|{nt}"].append(eid)
                        
            if name_norm:
                ngram_list = list(ngrams)
                ngram_list.sort(key=lambda ng: self.get_idf(self.name_ngram_df, ng), reverse=True)
                for ng in ngram_list[:self.num_ngrams]:
                    self.ngram_idx[ng].append(eid)
                    
            unique_addr_tokens = [t for t in set(addr_tokens) if not t.isdigit()]
            if unique_name_tokens and unique_addr_tokens:
                unique_addr_tokens.sort(key=lambda t: self.get_idf(self.address_token_df, t), reverse=True)
                top_nt = unique_name_tokens[0]
                top_at = unique_addr_tokens[0]
                self.name_addr_idx[f"{top_nt}|{top_at}"].append(eid)
                    
    def finalize_indexes(self):
        def cap_index(idx, limit):
            for k in idx:
                if len(idx[k]) > limit:
                    idx[k] = idx[k][:limit]
                    
        cap_index(self.global_token_idx, self.max_postings)
        cap_index(self.country_token_idx, self.max_postings)
        cap_index(self.ngram_idx, self.max_postings)
        cap_index(self.postal_token_idx, self.max_postings)
        cap_index(self.name_addr_idx, self.max_postings)
        cap_index(self.exact_name_idx, self.max_postings)

    def retrieve_candidates(self, row, strategies=None):
        candidates = set()
        
        name_norm = getattr(row, 'name_norm', '')
        if pd.isna(name_norm): name_norm = ''
        
        country = getattr(row, 'country', '')
        if pd.isna(country): country = ''
        country = country.lower()
        
        postal = getattr(row, 'postal_code', '')
        if pd.isna(postal): postal = ''
        
        nt_raw = getattr(row, 'name_tokens', [])
        if isinstance(nt_raw, str):
            nt_raw = ast.literal_eval(nt_raw) if nt_raw.startswith('[') else []
        name_tokens = [t for t in nt_raw if len(t) > 1]
        
        at_raw = getattr(row, 'address_tokens', [])
        if isinstance(at_raw, str):
            at_raw = ast.literal_eval(at_raw) if at_raw.startswith('[') else []
        addr_tokens = [t for t in at_raw if len(t) > 1]
        
        if 'A' in strategies and name_norm:
            candidates.update(self.exact_name_idx.get(name_norm, []))
            
        unique_name_tokens = list(set(name_tokens))
        if unique_name_tokens:
            unique_name_tokens.sort(key=lambda t: self.get_idf(self.name_token_df, t), reverse=True)
            
        if 'B' in strategies and unique_name_tokens:
            for nt in unique_name_tokens[:3]:
                if country:
                    candidates.update(self.country_token_idx.get(f"{country}|{nt}", []))
                else:
                    candidates.update(self.global_token_idx.get(nt, []))
                
        if 'C' in strategies and name_norm:
            ngrams = list(set(get_3grams(name_norm)))
            ngrams.sort(key=lambda ng: self.get_idf(self.name_ngram_df, ng), reverse=True)
            for ng in ngrams[:self.num_ngrams]:
                candidates.update(self.ngram_idx.get(ng, []))
                
        if 'D' in strategies and postal and unique_name_tokens:
            for nt in unique_name_tokens[:3]:
                candidates.update(self.postal_token_idx.get(f"{postal}|{nt}", []))
                
        unique_addr_tokens = [t for t in set(addr_tokens) if not t.isdigit()]
        if unique_addr_tokens:
            unique_addr_tokens.sort(key=lambda t: self.get_idf(self.address_token_df, t), reverse=True)
            
        if 'E' in strategies and unique_name_tokens and unique_addr_tokens:
            for nt in unique_name_tokens[:2]:
                for at in unique_addr_tokens[:2]:
                    candidates.update(self.name_addr_idx.get(f"{nt}|{at}", []))
                    
        return candidates

    def score_candidate_fast(self, c_eid, s1_meta):
        cand = self.store[c_eid]
        
        overlap = s1_meta['name_tokens'].intersection(cand['name_tokens'])
        name_score_idf = sum(self.get_idf(self.name_token_df, t) for t in overlap)
        
        denom = max(s1_meta['name_weight'], cand['name_weight'])
        norm_weighted_name_overlap = name_score_idf / denom if denom > 0 else 0.0
        
        addr_overlap = s1_meta['addr_tokens'].intersection(cand['addr_tokens'])
        addr_score_idf = sum(self.get_idf(self.address_token_df, t) for t in addr_overlap)
        
        addr_denom = max(s1_meta['addr_weight'], cand['addr_weight'])
        norm_weighted_addr_overlap = addr_score_idf / addr_denom if addr_denom > 0 else 0.0
        
        country_match = 1.0 if (s1_meta['country'] and cand['country'] and s1_meta['country'] == cand['country']) else 0.0
        postal_match = 1.0 if (s1_meta['postal'] and cand['postal'] and s1_meta['postal'] == cand['postal']) else 0.0
        
        c_ngrams = cand['ngrams']
        s1_ngrams = s1_meta['ngrams']
        if s1_ngrams and c_ngrams:
            ngram_jaccard = len(s1_ngrams.intersection(c_ngrams)) / len(s1_ngrams.union(c_ngrams))
        else:
            ngram_jaccard = 0.0
            
        final_score = name_score_idf + 5.0 * norm_weighted_name_overlap + 0.5 * addr_score_idf + 2.0 * norm_weighted_addr_overlap + 2.0 * country_match + 3.0 * postal_match + 5.0 * ngram_jaccard
        return final_score

    def rank_and_topk(self, s1_row, candidates, k):
        # Precompute S1 row metadata to save time
        nt_raw = getattr(s1_row, 'name_tokens', [])
        if isinstance(nt_raw, str): nt_raw = ast.literal_eval(nt_raw) if nt_raw.startswith('[') else []
        s1_name_tokens = set(nt_raw)
        
        at_raw = getattr(s1_row, 'address_tokens', [])
        if isinstance(at_raw, str): at_raw = ast.literal_eval(at_raw) if at_raw.startswith('[') else []
        s1_addr_tokens = set(at_raw)
        
        s1_name_weight = sum(self.get_idf(self.name_token_df, t) for t in s1_name_tokens)
        s1_addr_weight = sum(self.get_idf(self.address_token_df, t) for t in s1_addr_tokens)
        
        name_norm = getattr(s1_row, 'name_norm', '')
        if pd.isna(name_norm): name_norm = ''
        s1_ngrams = set(get_3grams(name_norm)) if name_norm else set()
        
        country = getattr(s1_row, 'country', '')
        if pd.isna(country): country = ''
        country = country.lower()
        
        postal = getattr(s1_row, 'postal_code', '')
        if pd.isna(postal): postal = ''
        
        s1_meta = {
            'name_tokens': s1_name_tokens,
            'addr_tokens': s1_addr_tokens,
            'name_weight': s1_name_weight,
            'addr_weight': s1_addr_weight,
            'ngrams': s1_ngrams,
            'country': country,
            'postal': postal
        }
        
        scored = []
        for cid in candidates:
            score = self.score_candidate_fast(cid, s1_meta)
            scored.append((score, cid))
            
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for score, cid in scored[:k]]
