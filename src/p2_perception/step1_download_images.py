import os
import time
import pandas as pd
import requests
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────
CSV_PATH   = "data/annotated/final_labels.csv"
OUTPUT_DIR = "data/images"
TIMEOUT    = 25
HEADERS    = {"User-Agent": "GreenwashGuard/1.0 (student project)"}
# ────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} products\n")

success, failed = 0, []

for _, row in tqdm(df.iterrows(), total=len(df)):
    code     = str(row['code']).strip()
    out_path = os.path.join(OUTPUT_DIR, f"{code}.jpg")

    # Skip if already downloaded
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        success += 1
        continue

    downloaded = False

    # ── Method 1: Use OFF API to get the real image URL ──
    try:
        api_url  = f"https://world.openfoodfacts.org/api/v2/product/{code}.json?fields=code,image_front_url,image_front_small_url"
        resp     = requests.get(api_url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            data    = resp.json()
            product = data.get("product", {})
            img_url = (product.get("image_front_url") or
                       product.get("image_front_small_url") or "")
            if img_url:
                img_resp = requests.get(img_url, headers=HEADERS, timeout=TIMEOUT)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    with open(out_path, 'wb') as f:
                        f.write(img_resp.content)
                    success += 1
                    downloaded = True
        time.sleep(0.3)   # be polite to the API
    except Exception:
        pass

    # ── Method 2: Build AWS URL using raw image number ──
    if not downloaded:
        try:
            padded  = code.zfill(13)
            folder  = f"{padded[0:3]}/{padded[3:6]}/{padded[6:9]}/{padded[9:13]}"
            # Try common raw image numbers
            for img_num in ["1", "2", "3"]:
                aws_url  = f"https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data/{folder}/{img_num}.jpg"
                img_resp = requests.get(aws_url, headers=HEADERS, timeout=TIMEOUT)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    with open(out_path, 'wb') as f:
                        f.write(img_resp.content)
                    success += 1
                    downloaded = True
                    break
        except Exception:
            pass

    # ── Method 3: Use original URL from CSV ──
    if not downloaded:
        try:
            orig_url = str(row.get('image_front_url', '')).strip()
            if orig_url and orig_url.startswith("http"):
                img_resp = requests.get(orig_url, headers=HEADERS, timeout=TIMEOUT)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    with open(out_path, 'wb') as f:
                        f.write(img_resp.content)
                    success += 1
                    downloaded = True
        except Exception:
            pass

    if not downloaded:
        failed.append(code)

print(f"\n✅ Downloaded : {success}")
print(f"❌ Failed     : {len(failed)}")

if failed:
    pd.Series(failed).to_csv("outputs/failed_downloads.csv", index=False)
    print(f"Failed barcodes saved to outputs/failed_downloads.csv")
    print(f"\nIf failed > 100, run the script again — it auto-resumes.")
