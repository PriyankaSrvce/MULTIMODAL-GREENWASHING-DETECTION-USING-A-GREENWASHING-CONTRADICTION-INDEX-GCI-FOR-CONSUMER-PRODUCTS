# src/p2_perception/step1_batch_download.py
# Smart batch downloader - downloads in small batches with breaks
# to avoid Open Food Facts server throttling
# Run: python src\p2_perception\step1_batch_download.py

import pandas as pd
import requests
import os
import time
import random
from tqdm import tqdm

os.makedirs('data/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Load failed barcodes ───────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/annotated/final_labels.csv')

# Check which ones are already valid
already_done = set()
for f in os.listdir('data/images'):
    fpath = f'data/images/{f}'
    if os.path.getsize(fpath) > 5000:
        already_done.add(f.replace('.jpg', ''))

print(f"Already downloaded (valid): {len(already_done)}")

# Get remaining ones
remaining = []
for _, row in df.iterrows():
    try:
        barcode = str(int(float(row['code'])))
    except:
        barcode = str(row['code'])
    if barcode not in already_done:
        remaining.append((barcode, str(row['image_front_url']).strip()))

print(f"Remaining to download: {len(remaining)}")
print()

# ── Rotating User Agents (looks like different browsers) ──────────
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# ── Settings ───────────────────────────────────────────────────────
BATCH_SIZE   = 20     # download 20 images then take a break
BREAK_TIME   = 15     # wait 15 seconds between batches
MIN_DELAY    = 0.5    # minimum delay between each image
MAX_DELAY    = 2.0    # maximum delay between each image

def download_one(barcode, url, session):
    save_path = f'data/images/{barcode}.jpg'
    ua = random.choice(USER_AGENTS)
    session.headers.update({'User-Agent': ua})
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200 and len(r.content) > 5000:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

# ── Main Download Loop ─────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://world.openfoodfacts.org/',
})

success = 0
failed  = 0
failed_barcodes = []
total   = len(remaining)
batch_num = 0

print(f"Starting batch download ({BATCH_SIZE} images per batch, {BREAK_TIME}s break between batches)")
print(f"Estimated time: {(total * 1.2 + (total // BATCH_SIZE) * BREAK_TIME) / 60:.0f}-{(total * 2.5 + (total // BATCH_SIZE) * BREAK_TIME) / 60:.0f} minutes")
print()

for i, (barcode, url) in enumerate(tqdm(remaining, desc="Downloading")):
    # Batch break
    if i > 0 and i % BATCH_SIZE == 0:
        batch_num += 1
        real_so_far = sum(
            1 for f in os.listdir('data/images')
            if os.path.getsize(f'data/images/{f}') > 5000
        )
        tqdm.write(f"\n--- Batch {batch_num} done | Real images so far: {real_so_far} | Pausing {BREAK_TIME}s ---\n")
        time.sleep(BREAK_TIME)

    if download_one(barcode, url, session):
        success += 1
    else:
        failed += 1
        failed_barcodes.append(barcode)

    # Random delay between requests
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

# ── Final count ────────────────────────────────────────────────────
real_total = sum(
    1 for f in os.listdir('data/images')
    if os.path.getsize(f'data/images/{f}') > 5000
)

print(f"\n{'='*55}")
print(f"FINAL REAL VALID IMAGE COUNT: {real_total} / 500")
print(f"Newly downloaded this run:    {success}")
print(f"Failed this run:              {failed}")
print(f"{'='*55}")

# Save failed list
if failed_barcodes:
    pd.DataFrame({'barcode': failed_barcodes}).to_csv('outputs/failed_downloads.csv', index=False)
    print(f"Failed barcodes saved to outputs/failed_downloads.csv")

if real_total >= 300:
    print(f"\n✅ {real_total} images — you are good to go!")
    print("   Next: python src\\p2_perception\\step2_clip_scorer.py")
elif real_total >= 200:
    print(f"\n⚠  {real_total} images — decent but you can run this script once more to get more.")
else:
    print(f"\n⚠  Only {real_total} images. Run again or contact Claude for hybrid approach.")
