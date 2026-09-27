import math
import collections
import numpy as np
import scipy.sparse as sp

def get_char_wb_ngrams(text, min_n=2, max_n=4):
    """Generate character n-grams with word boundaries."""
    ngrams = []
    if not text: 
        return ngrams
    words = text.split()
    for w in words:
        padded = " " + w + " "
        n_len = len(padded)
        for n in range(min_n, max_n + 1):
            for i in range(n_len - n + 1):
                ngrams.append(padded[i:i+n])
    return ngrams

class BlockingStrategiesV2_3:
    def __init__(self, max_postings=10000, name_ngram_range=(2,4)):
        self.max_postings = max_postings
        self.min_n, self.max_n = name_ngram_range
        
        self.doc_freq = collections.Counter()
        self.idf = {}
        self.total_docs = 0
        
        self.vocab = {}
        self.vocab_size = 0
        
        self.entity_ids = []
        self.country_ids = []
        
        self.country_map = {'unknown': 0}
        self.country_count = 1
        
        self.row_inds = []
        self.col_inds = []
        self.data_vals = []
        
        self.matrix = None

    def _get_country_id(self, country_str):
        if not country_str:
            return 0
        c = str(country_str).lower().strip()
        if c == 'nan' or c == '':
            return 0
        if c not in self.country_map:
            self.country_map[c] = self.country_count
            self.country_count += 1
        return self.country_map[c]

    def _get_vocab_id(self, feat_str):
        if feat_str not in self.vocab:
            self.vocab[feat_str] = self.vocab_size
            self.vocab_size += 1
        return self.vocab[feat_str]

    def _extract_features(self, row):
        name_norm = getattr(row, 'name_norm', '')
        addr_norm = getattr(row, 'address_norm', '')
        
        name_ngrams = get_char_wb_ngrams(name_norm, self.min_n, self.max_n)
        addr_tokens = str(addr_norm).split() if addr_norm else []
        
        features = set()
        
        for ng in name_ngrams:
            features.add(f"N_{ng}")
            
        for at in addr_tokens:
            features.add(f"A_{at}")
            
        # Add exact name match feature as a strong signal
        if name_norm:
            features.add(f"E_{name_norm}")
            
        return features

    def fit_frequencies(self, df):
        """First pass to compute document frequencies."""
        for row in df.itertuples(index=False):
            self.total_docs += 1
            features = self._extract_features(row)
            for feat in features:
                self.doc_freq[feat] += 1

    def build_indexes(self, df):
        """Second pass to build sparse matrix."""
        # Precompute IDFs
        for feat, count in self.doc_freq.items():
            if count <= self.max_postings:
                self.idf[feat] = math.log(self.total_docs / (count + 1))
        
        for row in df.itertuples(index=False):
            entity_idx = len(self.entity_ids)
            self.entity_ids.append(getattr(row, 'entity_id'))
            self.country_ids.append(self._get_country_id(getattr(row, 'country', '')))
            
            features = self._extract_features(row)
            
            for feat in features:
                if self.doc_freq[feat] <= self.max_postings:
                    # Weight components
                    weight = self.idf[feat]
                    if feat.startswith("E_"):
                        weight *= 5.0  # Boost exact name matches
                    elif feat.startswith("A_"):
                        weight *= 0.5  # Address is secondary to name
                        
                    v_id = self._get_vocab_id(feat)
                    self.row_inds.append(v_id)
                    self.col_inds.append(entity_idx)
                    self.data_vals.append(weight)

    def finalize_indexes(self):
        """Convert buffers to CSR sparse matrix."""
        print(f"V2.3 Vocab size: {self.vocab_size}, Total NNZ: {len(self.data_vals)}")
        self.matrix = sp.csr_matrix(
            (self.data_vals, (self.row_inds, self.col_inds)),
            shape=(self.vocab_size, len(self.entity_ids)),
            dtype=np.float32
        )
        
        self.row_inds = None
        self.col_inds = None
        self.data_vals = None
        self.doc_freq = None
        self.idf = None # We don't need it if we assume queries use uniform weights (1.0), but wait, we need idf for query terms too?
        # Actually in retrieve_and_score, we just put 1.0 in query, and M has the IDF weights.
        
        self.country_ids = np.array(self.country_ids, dtype=np.int32)

    def retrieve_and_score_batch(self, s1_chunk, k=50):
        """Retrieves and scores candidates for a batch of query rows using vectorized sparse dot product."""
        q_data = []
        q_row = []
        q_col = []
        
        country_ids = []
        
        for idx, s1_row in enumerate(s1_chunk):
            c_country_id = self._get_country_id(getattr(s1_row, 'country', ''))
            country_ids.append(c_country_id)
            
            features = self._extract_features(s1_row)
            
            for feat in features:
                v_id = self.vocab.get(feat)
                if v_id is not None:
                    q_row.append(idx)
                    q_col.append(v_id)
                    q_data.append(1.0)
                    
        batch_size = len(s1_chunk)
        if not q_data:
            return [[] for _ in range(batch_size)]
            
        Q = sp.csr_matrix(
            (q_data, (q_row, q_col)),
            shape=(batch_size, self.vocab_size),
            dtype=np.float32
        )
        
        # Sparse dot product
        scores_sparse = Q.dot(self.matrix)
        
        batch_results = []
        for i in range(batch_size):
            c_country_id = country_ids[i]
            row_start = scores_sparse.indptr[i]
            row_end = scores_sparse.indptr[i+1]
            
            c_cols = scores_sparse.indices[row_start:row_end]
            c_vals = scores_sparse.data[row_start:row_end]
            
            if len(c_cols) == 0:
                batch_results.append([])
                continue
                
            # Country filtering: match country OR either side is unknown
            # Unknown is country ID 0
            if c_country_id != 0:
                c_mask = (self.country_ids[c_cols] == c_country_id) | (self.country_ids[c_cols] == 0)
                valid_cols = c_cols[c_mask]
                valid_vals = c_vals[c_mask]
            else:
                valid_cols = c_cols
                valid_vals = c_vals
                
            if len(valid_cols) == 0:
                batch_results.append([])
                continue
                
            if len(valid_cols) > k:
                top_indices = np.argpartition(valid_vals, -k)[-k:]
                top_indices = top_indices[np.argsort(-valid_vals[top_indices])]
                top_cols = valid_cols[top_indices]
                top_vals = valid_vals[top_indices]
            else:
                sort_idx = np.argsort(-valid_vals)
                top_cols = valid_cols[sort_idx]
                top_vals = valid_vals[sort_idx]
                
            res = [(self.entity_ids[idx], float(val)) for idx, val in zip(top_cols, top_vals)]
            batch_results.append(res)
            
        return batch_results
