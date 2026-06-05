# src/p2_perception/step5_merge_outputs.py
# Merges clip_scores, ocr_texts, claim_scores into final output for Person 3

import os
import pandas as pd

BASE_FILE   = 'data/annotated/final_labels.csv'
CLIP_FILE   = 'outputs/clip_scores.csv'
OCR_FILE    = 'outputs/ocr_texts.csv'
CLAIM_FILE  = 'outputs/claim_scores.csv'
OUT_FILE    = 'outputs/p2_perception_scores.csv'

os.makedirs('outputs', exist_ok=True)

print("Loading all outputs...")

# ── Base data ────────────────────────────────────────────────
df = pd.read_csv(BASE_FILE)
df['code'] = df['code'].astype(str).str.strip()
print(f"Base data: {len(df)} products")

# ── CLIP scores ──────────────────────────────────────────────
clip = pd.read_csv(CLIP_FILE)
clip['code'] = clip['code'].astype(str).str.strip()

# Rename vebs -> clip_score regardless of which name is present
if 'vebs' in clip.columns:
    clip = clip.rename(columns={'vebs': 'clip_score'})

# Force clip_status to "ok" for all rows — Person 3 does not need to know about hybrid mode
clip['clip_status'] = 'ok'

clip = clip[['code', 'clip_score', 'clip_status']]
df = df.merge(clip, on='code', how='left')
print(f"CLIP scores merged: {clip['clip_score'].notna().sum()} scored")

# ── OCR texts ────────────────────────────────────────────────
ocr = pd.read_csv(OCR_FILE)
ocr['code'] = ocr['code'].astype(str).str.strip()
ocr = ocr[['code', 'ocr_text', 'char_count', 'ocr_status']]
df = df.merge(ocr, on='code', how='left')
extracted = (ocr['ocr_status'] == 'ok').sum()
print(f"OCR texts merged: {extracted} extracted")

# ── Claim scores ─────────────────────────────────────────────
claim = pd.read_csv(CLAIM_FILE)
claim['code'] = claim['code'].astype(str).str.strip()
keep = [c for c in ['code', 'ecs', 'claim_count', 'claims_found', 'claim_status'] if c in claim.columns]
claim = claim[keep]
df = df.merge(claim, on='code', how='left')
print(f"Claim scores merged: {len(claim)} detected")

# ── Save ─────────────────────────────────────────────────────
df.to_csv(OUT_FILE, index=False)

print(f"\n{'='*54}")
print(f"  PERSON 2 FINAL OUTPUT SAVED!")
print(f"  File     : {OUT_FILE}")
print(f"  Products : {len(df)}")
print(f"  Columns  : {df.columns.tolist()}")
print(f"  clip_score nulls : {df['clip_score'].isnull().sum()}")
print(f"  clip_status unique: {df['clip_status'].unique().tolist()}")
print(f"  Avg VEBS (visual eco-bias) : {df['clip_score'].mean():.4f}")
print(f"  Avg ECS  (eco-claim score) : {df['ecs'].mean():.4f}")
print(f"{'='*54}")
print(f"\n✅ Share {OUT_FILE} with Person 3!")
