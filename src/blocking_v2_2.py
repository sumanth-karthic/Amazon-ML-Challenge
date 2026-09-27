import time
import math
import collections
import numpy as np
import scipy.sparse as sp

class BlockingStrategiesV2_2:
    def __init__(self, max_postings=500, num_ngrams=5):
        self.max_postings = max_postings
        self.num_ngrams = num_ngrams
        
        self.doc_freq = collections.Counter()
        self.idf = {}
        self.total_docs = 0
        
        self.vocab = {}  # feature_string -> int_id
        self.vocab_size = 0
        
        self.entity_ids = []      # list of string entity_ids
        self.country_ids = []     # list of int country_ids
        
        # We will map country strings to ints
        self.country_map = {'unknown': 0}
        self.country_count = 1
        
        # Sparse matrix construction buffers
        self.row_inds = []
        self.col_inds = []
        self.data_vals = []
        
        self.matrix = None

    def _get_country_id(self, country_str):
        if not country_str:
            return 0
        c = country_str.lower()
        if c not in self.country_map:
            self.country_map[c] = self.country_count
            self.country_count += 1
        return self.country_map[c]

    def _get_vocab_id(self, feat_str):
        if feat_str not in self.vocab:
            self.vocab[feat_str] = self.vocab_size
            self.vocab_size += 1
        return self.vocab[feat_str]

    def fit_frequencies(self, df):
        """First pass to compute frequencies."""
        for row in df.itertuples(index=False):
            self.total_docs += 1
            name_tokens = getattr(row, 'name_norm', '').split()
            addr_tokens = getattr(row, 'address_norm', '').split()
            ngrams = getattr(row, 'ngrams', [])
            postal = getattr(row, 'postal_code', '')
            
            seen = set()
            for nt in name_tokens:
                seen.add(f"N:{nt}")
            for at in addr_tokens:
                seen.add(f"A:{at}")
            for ng in ngrams[:self.num_ngrams]:
                seen.add(f"G:{ng}")
                
            # Combinatorial retrieval features (used to emulate V2.1 Strategy D & E)
            if postal:
                for nt in name_tokens[:3]:
                    seen.add(f"P:{postal}|{nt}")
            
            for nt in name_tokens[:2]:
                for at in addr_tokens[:2]:
                    seen.add(f"NA:{nt}|{at}")
                    
            for feat in seen:
                self.doc_freq[feat] += 1

    def build_indexes(self, df):
        """Second pass to build sparse matrix."""
        # Calculate IDFs for name and address tokens
        for feat, count in self.doc_freq.items():
            if feat.startswith("N:") or feat.startswith("A:"):
                raw_token = feat[2:]
                if raw_token not in self.idf:
                    # Same IDF formula as V2.1
                    self.idf[raw_token] = math.log(self.total_docs / (count + 1))
        
        for row in df.itertuples(index=False):
            entity_idx = len(self.entity_ids)
            self.entity_ids.append(getattr(row, 'entity_id'))
            
            country_id = self._get_country_id(getattr(row, 'country', ''))
            self.country_ids.append(country_id)
            
            name_tokens = getattr(row, 'name_norm', '').split()
            addr_tokens = getattr(row, 'address_norm', '').split()
            ngrams = getattr(row, 'ngrams', [])
            postal = getattr(row, 'postal_code', '')
            exact = getattr(row, 'name_norm', '')
            
            features = {}
            
            if exact:
                features[f"E:{exact}"] = 10.0
                
            for nt in name_tokens:
                feat = f"N:{nt}"
                if self.doc_freq[feat] <= self.max_postings:
                    features[feat] = self.idf.get(nt, 0.5)
                    
            for at in addr_tokens:
                feat = f"A:{at}"
                if self.doc_freq[feat] <= self.max_postings:
                    features[feat] = 0.5 * self.idf.get(at, 0.5)
                    
            for ng in ngrams[:self.num_ngrams]:
                feat = f"G:{ng}"
                if self.doc_freq[feat] <= self.max_postings:
                    features[feat] = 0.1
                    
            if postal:
                for nt in name_tokens[:3]:
                    feat = f"P:{postal}|{nt}"
                    if self.doc_freq[feat] <= self.max_postings:
                        features[feat] = 0.0001
                        
            for nt in name_tokens[:2]:
                for at in addr_tokens[:2]:
                    feat = f"NA:{nt}|{at}"
                    if self.doc_freq[feat] <= self.max_postings:
                        features[feat] = 0.0001
            
            for feat, weight in features.items():
                v_id = self._get_vocab_id(feat)
                # We build Matrix M of shape (Vocab, Entities)
                self.row_inds.append(v_id)
                self.col_inds.append(entity_idx)
                self.data_vals.append(weight)

    def finalize_indexes(self):
        """Convert buffers to CSR sparse matrix."""
        print(f"Vocab size: {self.vocab_size}, Total NNZ: {len(self.data_vals)}")
        self.matrix = sp.csr_matrix(
            (self.data_vals, (self.row_inds, self.col_inds)),
            shape=(self.vocab_size, len(self.entity_ids)),
            dtype=np.float32
        )
        
        # Clear buffers to free memory
        self.row_inds = None
        self.col_inds = None
        self.data_vals = None
        self.doc_freq = None
        
        self.country_ids = np.array(self.country_ids, dtype=np.int32)

    def retrieve_and_score(self, s1_row, k=50):
        """Retrieves and scores candidates using a single sparse dot product."""
        # Extract features for query
        name_tokens = getattr(s1_row, 'name_norm', '').split()
        addr_tokens = getattr(s1_row, 'address_norm', '').split()
        ngrams = getattr(s1_row, 'ngrams', [])
        postal = getattr(s1_row, 'postal_code', '')
        exact = getattr(s1_row, 'name_norm', '')
        
        q_cols = []
        q_vals = []
        
        # Build query vector Q. We map features to vocab_id.
        # Note: in V2.1 scoring, Q values are 1.0, and M values are the weights.
        features = set()
        if exact:
            features.add(f"E:{exact}")
        for nt in name_tokens:
            features.add(f"N:{nt}")
        for at in addr_tokens:
            features.add(f"A:{at}")
        for ng in ngrams[:self.num_ngrams]:
            features.add(f"G:{ng}")
            
        if postal:
            for nt in name_tokens[:3]:
                features.add(f"P:{postal}|{nt}")
        for nt in name_tokens[:2]:
            for at in addr_tokens[:2]:
                features.add(f"NA:{nt}|{at}")
                
        for feat in features:
            v_id = self.vocab.get(feat)
            if v_id is not None:
                q_cols.append(v_id)
                q_vals.append(1.0)
                
        if not q_cols:
            return []
            
        # Create sparse row vector Q (shape: 1 x Vocab)
        Q = sp.csr_matrix(
            (q_vals, ([0]*len(q_cols), q_cols)),
            shape=(1, self.vocab_size),
            dtype=np.float32
        )
        
        # Sparse dot product Q * M (shape: 1 x Entities)
        scores_sparse = Q.dot(self.matrix)
        
        c_cols = scores_sparse.indices
        c_vals = scores_sparse.data
        
        if len(c_cols) == 0:
            return []
            
        # Filter by country
        c_country_id = self._get_country_id(getattr(s1_row, 'country', ''))
        
        if c_country_id != 0:
            c_mask = (self.country_ids[c_cols] == c_country_id) | (self.country_ids[c_cols] == 0)
            valid_cols = c_cols[c_mask]
            valid_vals = c_vals[c_mask]
        else:
            valid_cols = c_cols
            valid_vals = c_vals
            
        if len(valid_cols) == 0:
            return []
            
        # Top-K selection
        if len(valid_cols) > k:
            # np.argpartition is O(N)
            top_indices = np.argpartition(valid_vals, -k)[-k:]
            # Sort the top k descending
            top_indices = top_indices[np.argsort(-valid_vals[top_indices])]
            
            top_cols = valid_cols[top_indices]
            top_vals = valid_vals[top_indices]
        else:
            # Sort descending
            sort_idx = np.argsort(-valid_vals)
            top_cols = valid_cols[sort_idx]
            top_vals = valid_vals[sort_idx]
            
        # Map back to entity IDs
        results = [(self.entity_ids[idx], float(val)) for idx, val in zip(top_cols, top_vals)]
        return results

    def retrieve_and_score_batch(self, s1_chunk, k=50):
        """Retrieves and scores candidates for a batch of query rows using vectorized sparse dot product."""
        q_data = []
        q_row = []
        q_col = []
        
        country_ids = []
        
        for idx, s1_row in enumerate(s1_chunk):
            name_tokens = getattr(s1_row, 'name_norm', '').split()
            addr_tokens = getattr(s1_row, 'address_norm', '').split()
            ngrams = getattr(s1_row, 'ngrams', [])
            postal = getattr(s1_row, 'postal_code', '')
            exact = getattr(s1_row, 'name_norm', '')
            
            c_country_id = self._get_country_id(getattr(s1_row, 'country', ''))
            country_ids.append(c_country_id)
            
            features = set()
            if exact: features.add(f"E:{exact}")
            for nt in name_tokens: features.add(f"N:{nt}")
            for at in addr_tokens: features.add(f"A:{at}")
            for ng in ngrams[:self.num_ngrams]: features.add(f"G:{ng}")
                
            if postal:
                for nt in name_tokens[:3]: features.add(f"P:{postal}|{nt}")
            for nt in name_tokens[:2]:
                for at in addr_tokens[:2]: features.add(f"NA:{nt}|{at}")
                    
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
