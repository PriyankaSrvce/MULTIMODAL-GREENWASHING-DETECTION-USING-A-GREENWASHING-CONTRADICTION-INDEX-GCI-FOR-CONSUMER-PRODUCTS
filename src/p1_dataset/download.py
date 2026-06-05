import requests, time, pandas as pd, os

os.makedirs('data/raw', exist_ok=True)

def fetch_products(category, page=1):
    try:
        url = 'https://world.openfoodfacts.org/cgi/search.pl'
        params = {
            'action': 'process',
            'tagtype_0': 'categories',
            'tag_contains_0': 'contains',
            'tag_0': category,
            'json': '1',
            'page_size': '100',
            'page': str(page),
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get('products', [])
    except Exception as e:
        print(f'  Skipping page {page} of {category}: {e}')
        return []

CATEGORIES = [
    'biscuits-and-cakes', 'breakfast-cereals',
    'juices-and-nectars', 'snacks', 'dairy-desserts',
    'energy-drinks', 'chocolates', 'chips-and-crisps'
]

all_products = []
for cat in CATEGORIES:
    print(f'Fetching: {cat}')
    for page in range(1, 4):
        products = fetch_products(cat, page)
        if not products:
            break
        all_products.extend(products)
        time.sleep(2)
    print(f'  Total so far: {len(all_products)}')

df = pd.DataFrame(all_products)
df.to_parquet('data/raw/off_api_sample.parquet', index=False)
print(f'Saved {len(df)} products!')