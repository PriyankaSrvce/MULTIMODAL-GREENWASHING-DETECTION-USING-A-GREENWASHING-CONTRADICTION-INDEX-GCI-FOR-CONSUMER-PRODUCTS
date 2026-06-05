import pandas as pd
import os

os.makedirs('data/filtered', exist_ok=True)

print('Reading real Open Food Facts data...')

# Read only the columns we need from the TSV file
COLS = [
    'code', 'product_name', 'brands', 'countries_en',
    'ingredients_text', 'labels_tags', 'categories_en',
    'image_front_url', 'additives_tags', 'nova_group'
]

df = pd.read_csv(
    'data/raw/en.openfoodfacts.org.products.tsv',
    sep='\t',
    usecols=lambda c: c in COLS,
    low_memory=False,
    nrows=50000  # read first 50k rows only
)

print(f'Loaded {len(df):,} products')

# Keep only products with required fields
df = df.dropna(subset=['code', 'product_name', 'ingredients_text'])
df = df[df['ingredients_text'].str.strip().str.len() > 30]
print(f'After cleaning: {len(df):,}')

# Keep products with eco/health claims
claim_keywords = [
    'natural', 'organic', 'no added sugar', 'sugar free',
    'healthy', 'pure', 'wholesome', 'no preservatives',
    'gluten free', 'vegan'
]

def has_claim(row):
    text = str(row.get('product_name', '')) + ' ' + str(row.get('labels_tags', ''))
    text = text.lower()
    return any(kw in text for kw in claim_keywords)

with_claims = df[df.apply(has_claim, axis=1)]
without_claims = df[~df.apply(has_claim, axis=1)].sample(
    min(100, len(df)), random_state=42)

df_filtered = pd.concat([with_claims, without_claims]).drop_duplicates('code')
print(f'After claim filter: {len(df_filtered):,}')

# Sample to 500
if len(df_filtered) > 500:
    df_filtered = df_filtered.sample(500, random_state=42)

df_filtered['greenwash_label'] = ''
df_filtered['annotation_notes'] = ''

df_filtered.to_parquet('data/filtered/off_filtered.parquet', index=False)
df_filtered.to_csv('data/filtered/off_filtered_for_annotation.csv', index=False)

print(f'Done! Saved {len(df_filtered)} products')
print('Check data/filtered/off_filtered_for_annotation.csv')