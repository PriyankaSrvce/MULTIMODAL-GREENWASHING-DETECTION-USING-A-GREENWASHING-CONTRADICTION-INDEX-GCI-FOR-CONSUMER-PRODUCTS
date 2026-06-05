#!/usr/bin/env python3
"""
run_person2.py
==============
Person 2's master script — runs all 4 perception steps in order.

Usage:
    python run_person2.py

Steps:
    Step 1 — Download images from URLs
    Step 2 — CLIP visual scoring (VEBS)
    Step 3 — OCR text extraction
    Step 4 — DeBERTa claim detection (ECS)
    Step 5 — Merge all scores and save final Person 2 output

Output:
    outputs/p2_perception_scores.csv  ← hand this to Person 3
"""

import os, sys, pandas as pd

# ── Make sure we can import from src/ ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs('outputs', exist_ok=True)
os.makedirs('data/images', exist_ok=True)

print('='*60)
print('  GREENWASH GUARD — Person 2 Perception Pipeline')
print('='*60)

# ────────────────────────────────────────────────────────────────────
# STEP 1: Download images
# ────────────────────────────────────────────────────────────────────
print('\n[STEP 1/4] Downloading product images...')
from src.p2_perception.download_images import *

df = pd.read_csv('data/annotated/final_labels.csv')
df_imgs = df.dropna(subset=['image_front_url'])
success = 0
failed  = 0
for _, row in df_imgs.iterrows():
    barcode   = str(int(row['code'])) if not pd.isna(row['code']) else 'unknown'
    url       = str(row['image_front_url'])
    save_path = f'data/images/{barcode}.jpg'
    if download_image(url, save_path):
        success += 1
    else:
        failed += 1

print(f'  Downloaded {success} images, {failed} failed.')

# ────────────────────────────────────────────────────────────────────
# STEP 2: CLIP Visual Scoring
# ────────────────────────────────────────────────────────────────────
print('\n[STEP 2/4] Running CLIP visual scoring (VEBS)...')
from src.p2_perception.clip_scorer import run_clip_on_all
clip_df = run_clip_on_all()

# ────────────────────────────────────────────────────────────────────
# STEP 3: OCR Text Extraction
# ────────────────────────────────────────────────────────────────────
print('\n[STEP 3/4] Running OCR text extraction...')
from src.p2_perception.ocr_extractor import run_ocr_on_all
ocr_df = run_ocr_on_all()

# ────────────────────────────────────────────────────────────────────
# STEP 4: Claim Detection
# ────────────────────────────────────────────────────────────────────
print('\n[STEP 4/4] Running DeBERTa claim detection (ECS)...')
from src.p2_perception.claim_detector import run_claim_detection_on_all
claim_df = run_claim_detection_on_all()

# ────────────────────────────────────────────────────────────────────
# STEP 5: Merge everything into one output file for Person 3
# ────────────────────────────────────────────────────────────────────
print('\n[FINAL] Merging all scores...')

base_df = pd.read_csv('data/annotated/final_labels.csv')[
    ['code', 'product_name', 'brands', 'ingredients_text',
     'labels_tags', 'greenwash_label']
]

# Merge CLIP scores
merged = base_df.merge(
    clip_df[['code', 'vebs', 'eco_similarity', 'counter_similarity']],
    on='code', how='left'
)

# Merge OCR results
merged = merged.merge(
    ocr_df[['code', 'ocr_text', 'ocr_confidence', 'ocr_word_count']],
    on='code', how='left'
)

# Merge claim scores
merged = merged.merge(
    claim_df[['code', 'ecs', 'claim_count', 'claims_detected', 'category_scores']],
    on='code', how='left'
)

# Fill nulls with neutral defaults
merged['vebs']  = merged['vebs'].fillna(0.5)
merged['ecs']   = merged['ecs'].fillna(0.0)
merged['claim_count'] = merged['claim_count'].fillna(0)

out_path = 'outputs/p2_perception_scores.csv'
merged.to_csv(out_path, index=False)

print('\n' + '='*60)
print('  PERSON 2 PIPELINE COMPLETE!')
print('='*60)
print(f'\n  Output saved to: {out_path}')
print(f'  Total products processed: {len(merged)}')
print(f'  Columns: {list(merged.columns)}')
print('\n  Share outputs/p2_perception_scores.csv with Person 3.')
print('\nSample results:')
print(merged[['product_name', 'vebs', 'ecs', 'claim_count']].head(10).to_string(index=False))
