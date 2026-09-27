import pandas as pd
import unicodedata
import re

import sys

# Build translation table for punctuation and symbols mapping to space
PUNCT_DICT = {i: ' ' for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith(('P', 'S'))}

def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Unicode normalize
    text = unicodedata.normalize('NFC', text)
    # casefold
    text = text.casefold()
    # Normalize common ampersand usage
    text = text.replace('&', ' and ')
    
    # Replace punctuation and symbols with space safely
    text = text.translate(PUNCT_DICT)
    
    # Remove underscore manually if not covered (it is covered under 'Pc' - Connector Punctuation)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_postal_code(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Look for Indian (6 digits), US (5 or 5-4), France (5 digits)
    # It's better to search from the end or just find all and pick the last one.
    matches = re.findall(r'\b(?:\d{5}(?:-\d{4})?|\d{6})\b', text)
    if matches:
        return matches[-1]
    return ""

def preprocess_chunk(df):
    """
    Preprocess a pandas dataframe chunk.
    """
    # 3. Preserve original columns
    # Assumes df has ['entity_id', 'business_name', 'business_address', 'country']
    
    # 4 & 5. Create normalized name and address
    df['name_norm'] = df['business_name'].apply(normalize_text)
    df['address_norm'] = df['business_address'].apply(normalize_text)
    
    # 6. (Memory optimization: Do NOT store tokens as lists in DataFrame)
    # df['name_tokens'] = df['name_norm'].apply(lambda x: x.split() if x else [])
    # df['address_tokens'] = df['address_norm'].apply(lambda x: x.split() if x else [])
    
    # 7. Extract possible postal/PIN codes
    df['postal_code'] = df['business_address'].apply(extract_postal_code)
    
    # 8. Add useful boolean fields
    df['name_missing'] = df['name_norm'] == ""
    df['address_missing'] = df['address_norm'] == ""
    df['postal_missing'] = df['postal_code'] == ""
    
    return df

def preprocess_file(input_path, output_path, chunksize=100000, nrows=None):
    """
    Process large files in chunks.
    """
    chunk_iter = pd.read_csv(
        input_path, 
        sep='\t', 
        chunksize=chunksize, 
        nrows=nrows,
        dtype=str,
        quoting=3,
        on_bad_lines='warn'
    )
    
    first_chunk = True
    for chunk in chunk_iter:
        processed = preprocess_chunk(chunk)
        processed.to_csv(
            output_path, 
            sep='\t', 
            index=False, 
            mode='w' if first_chunk else 'a', 
            header=first_chunk
        )
        first_chunk = False
