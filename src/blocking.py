import collections

class FrequencyAwareIndex:
    def __init__(self, max_postings):
        self.max_postings = max_postings
        self.index = {}
        self.freq = collections.Counter()

    def add(self, key, entity_id):
        self.freq[key] += 1
        if self.freq[key] <= self.max_postings:
            if key not in self.index:
                self.index[key] = []
            self.index[key].append(entity_id)
        elif self.freq[key] == self.max_postings + 1:
            # Memory optimization: drop the list if it exceeds the limit
            del self.index[key]

    def get(self, key):
        return self.index.get(key, [])

class BlockingStrategies:
    def __init__(self, max_postings=1000):
        # Indexes
        self.exact_name_idx = FrequencyAwareIndex(max_postings)
        self.name_country_idx = FrequencyAwareIndex(max_postings)
        self.token_country_idx = FrequencyAwareIndex(max_postings)
        self.postal_token_idx = FrequencyAwareIndex(max_postings)
        self.name_addr_token_idx = FrequencyAwareIndex(max_postings)

    def fit_record(self, row):
        entity_id = row.get('entity_id')
        name_norm = row.get('name_norm', '')
        country = row.get('country', '').lower()
        postal = row.get('postal_code', '')
        name_tokens = row.get('name_tokens', [])
        addr_tokens = row.get('address_tokens', [])
        
        if name_norm:
            self.exact_name_idx.add(name_norm, entity_id)
            if country:
                self.name_country_idx.add(f"{name_norm}|{country}", entity_id)

        # Token + Country
        if country:
            for nt in set(name_tokens):
                if len(nt) > 1: # avoid single char tokens
                    self.token_country_idx.add(f"{nt}|{country}", entity_id)

        # Postal + Name token
        if postal:
            for nt in set(name_tokens):
                if len(nt) > 1:
                    self.postal_token_idx.add(f"{postal}|{nt}", entity_id)

        # Name token + Address token (limited to avoid explosion)
        # To avoid combinatorial explosion, only index if we have few tokens
        if len(name_tokens) <= 5 and len(addr_tokens) <= 10:
            for nt in set(name_tokens):
                if len(nt) > 1:
                    for at in set(addr_tokens):
                        if len(at) > 1 and not at.isdigit(): # avoid matching pure numbers for addr
                            self.name_addr_token_idx.add(f"{nt}|{at}", entity_id)

    def fit_dataframe(self, df):
        for _, row in df.iterrows():
            self.fit_record(row)

    def get_candidates_A(self, row):
        # A. Exact normalized name
        name = row.get('name_norm', '')
        if name:
            return set(self.exact_name_idx.get(name))
        return set()

    def get_candidates_B(self, row):
        # B. Name + country
        name = row.get('name_norm', '')
        country = row.get('country', '').lower()
        if name and country:
            return set(self.name_country_idx.get(f"{name}|{country}"))
        return set()

    def get_candidates_C(self, row):
        # C. Rare name token + country
        candidates = set()
        country = row.get('country', '').lower()
        if country:
            for nt in set(row.get('name_tokens', [])):
                if len(nt) > 1:
                    candidates.update(self.token_country_idx.get(f"{nt}|{country}"))
        return candidates

    def get_candidates_D(self, row):
        # D. Postal + name token
        candidates = set()
        postal = row.get('postal_code', '')
        if postal:
            for nt in set(row.get('name_tokens', [])):
                if len(nt) > 1:
                    candidates.update(self.postal_token_idx.get(f"{postal}|{nt}"))
        return candidates
        
    def get_candidates_E(self, row):
        # E. Name token + address token
        candidates = set()
        name_tokens = row.get('name_tokens', [])
        addr_tokens = row.get('address_tokens', [])
        
        # We query all combinations
        for nt in set(name_tokens):
            if len(nt) > 1:
                for at in set(addr_tokens):
                    if len(at) > 1 and not at.isdigit():
                        candidates.update(self.name_addr_token_idx.get(f"{nt}|{at}"))
        return candidates

    def get_candidates_F(self, row):
        # F. Union of strategies C and D (as an example of strongest)
        # Let's say union of A, B, C, D
        cands = set()
        cands.update(self.get_candidates_A(row))
        cands.update(self.get_candidates_B(row))
        cands.update(self.get_candidates_C(row))
        cands.update(self.get_candidates_D(row))
        return cands

