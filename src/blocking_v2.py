import collections
import math

def get_3grams(text):
    text = f"^{text}$"
    return [text[i:i+3] for i in range(len(text)-2)]

class BlockingStrategiesV2:
    def __init__(self, max_postings=5000):
        self.max_postings = max_postings
        
        # Frequencies
        self.token_df = collections.Counter()
        self.ngram_df = collections.Counter()
        self.total_docs = 0
        
        # Indexes
        self.exact_name_idx = collections.defaultdict(list)
        self.token_idx = collections.defaultdict(list)
        self.ngram_idx = collections.defaultdict(list)
        self.postal_token_idx = collections.defaultdict(list)
        self.name_addr_idx = collections.defaultdict(list)
        
        # Store for quick scoring
        self.store = {}
        
    def get_idf(self, df_counter, token):
        # Laplace smoothed IDF
        df = df_counter.get(token, 0)
        return math.log((self.total_docs + 1) / (df + 1))
        
    def fit_frequencies(self, df):
        for _, row in df.iterrows():
            name_tokens = set(row.get('name_tokens', []))
            for nt in name_tokens:
                if len(nt) > 1:
                    self.token_df[nt] += 1
                
            name_norm = row.get('name_norm', '')
            if name_norm:
                ngrams = set(get_3grams(name_norm))
                for ng in ngrams:
                    self.ngram_df[ng] += 1
                    
            # Address token freq
            addr_tokens = set(row.get('address_tokens', []))
            for at in addr_tokens:
                if len(at) > 1:
                    self.token_df[at] += 1
                    
            self.total_docs += 1

    def build_indexes(self, df):
        for _, row in df.iterrows():
            eid = row['entity_id']
            name_norm = row.get('name_norm', '')
            country = row.get('country', '').lower()
            postal = row.get('postal_code', '')
            name_tokens = [t for t in row.get('name_tokens', []) if len(t) > 1]
            addr_tokens = [t for t in row.get('address_tokens', []) if len(t) > 1]
            
            # Store metadata
            self.store[eid] = {
                'name_tokens': set(name_tokens),
                'addr_tokens': set(addr_tokens),
                'country': country,
                'postal': postal,
                'name_norm': name_norm
            }
            
            # 1. Exact name
            if name_norm:
                self.exact_name_idx[name_norm].append(eid)
                
            # 2. Top informative name tokens
            unique_name_tokens = list(set(name_tokens))
            if unique_name_tokens:
                unique_name_tokens.sort(key=lambda t: self.get_idf(self.token_df, t), reverse=True)
                # Keep top 2 tokens
                for nt in unique_name_tokens[:2]:
                    self.token_idx[nt].append(eid)
                    if postal:
                        self.postal_token_idx[f"{postal}|{nt}"].append(eid)
                        
            # 3. Top informative n-grams
            if name_norm:
                ngrams = list(set(get_3grams(name_norm)))
                ngrams.sort(key=lambda ng: self.get_idf(self.ngram_df, ng), reverse=True)
                for ng in ngrams[:3]: # top 3 rarest n-grams
                    self.ngram_idx[ng].append(eid)
                    
            # 4. Name + Address token (limited to rarest)
            unique_addr_tokens = [t for t in set(addr_tokens) if not t.isdigit()]
            if unique_name_tokens and unique_addr_tokens:
                unique_addr_tokens.sort(key=lambda t: self.get_idf(self.token_df, t), reverse=True)
                top_nt = unique_name_tokens[0]
                top_at = unique_addr_tokens[0]
                self.name_addr_idx[f"{top_nt}|{top_at}"].append(eid)
                    
    def finalize_indexes(self):
        # Cap posting lists to prevent combinatorial explosion during retrieval
        def cap_index(idx, limit):
            for k in list(idx.keys()):
                if len(idx[k]) > limit:
                    # Drop entirely if it's too common
                    del idx[k]
                    
        cap_index(self.token_idx, self.max_postings)
        cap_index(self.ngram_idx, self.max_postings)
        cap_index(self.postal_token_idx, self.max_postings)
        cap_index(self.name_addr_idx, self.max_postings)

    def retrieve_candidates(self, row, strategies=None):
        candidates = set()
        name_norm = row.get('name_norm', '')
        postal = row.get('postal_code', '')
        name_tokens = [t for t in row.get('name_tokens', []) if len(t) > 1]
        addr_tokens = [t for t in row.get('address_tokens', []) if len(t) > 1]
        
        # A. Exact name
        if 'A' in strategies and name_norm:
            candidates.update(self.exact_name_idx.get(name_norm, []))
            
        # B. Name tokens
        unique_name_tokens = list(set(name_tokens))
        if unique_name_tokens:
            unique_name_tokens.sort(key=lambda t: self.get_idf(self.token_df, t), reverse=True)
        if 'B' in strategies and unique_name_tokens:
            for nt in unique_name_tokens[:3]: # query top 3
                candidates.update(self.token_idx.get(nt, []))
                
        # C. N-grams
        if 'C' in strategies and name_norm:
            ngrams = list(set(get_3grams(name_norm)))
            ngrams.sort(key=lambda ng: self.get_idf(self.ngram_df, ng), reverse=True)
            for ng in ngrams[:5]: # query top 5
                candidates.update(self.ngram_idx.get(ng, []))
                
        # D. Postal + token
        if 'D' in strategies and postal and unique_name_tokens:
            for nt in unique_name_tokens[:3]:
                candidates.update(self.postal_token_idx.get(f"{postal}|{nt}", []))
                
        # E. Name token + addr token
        unique_addr_tokens = [t for t in set(addr_tokens) if not t.isdigit()]
        if unique_addr_tokens:
            unique_addr_tokens.sort(key=lambda t: self.get_idf(self.token_df, t), reverse=True)
        if 'E' in strategies and unique_name_tokens and unique_addr_tokens:
            for nt in unique_name_tokens[:2]:
                for at in unique_addr_tokens[:2]:
                    candidates.update(self.name_addr_idx.get(f"{nt}|{at}", []))
                    
        return candidates

    def score_candidate(self, s1_row, cand_eid):
        cand = self.store[cand_eid]
        
        s1_name_tokens = set(s1_row.get('name_tokens', []))
        c_name_tokens = cand['name_tokens']
        
        overlap = s1_name_tokens.intersection(c_name_tokens)
        name_score = sum(self.get_idf(self.token_df, t) for t in overlap)
        
        s1_addr_tokens = set(s1_row.get('address_tokens', []))
        c_addr_tokens = cand['addr_tokens']
        addr_overlap = s1_addr_tokens.intersection(c_addr_tokens)
        addr_score = sum(self.get_idf(self.token_df, t) for t in addr_overlap)
        
        country_match = 1.0 if (s1_row.get('country') and cand['country'] and s1_row.get('country').lower() == cand['country']) else 0.0
        postal_match = 1.0 if (s1_row.get('postal_code') and cand['postal'] and s1_row.get('postal_code') == cand['postal']) else 0.0
        
        # Optional: character n-gram Jaccard on exact name
        s1_ngrams = set(get_3grams(s1_row.get('name_norm', '')))
        c_ngrams = set(get_3grams(cand['name_norm']))
        if s1_ngrams and c_ngrams:
            ngram_jaccard = len(s1_ngrams.intersection(c_ngrams)) / len(s1_ngrams.union(c_ngrams))
        else:
            ngram_jaccard = 0.0
            
        return name_score + 0.5 * addr_score + 2.0 * country_match + 3.0 * postal_match + 5.0 * ngram_jaccard

    def rank_and_topk(self, s1_row, candidates, k):
        scored = []
        for cid in candidates:
            score = self.score_candidate(s1_row, cid)
            scored.append((score, cid))
            
        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for score, cid in scored[:k]]
