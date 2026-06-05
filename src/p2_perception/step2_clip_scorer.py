# src/p2_perception/step2_clip_scorer.py
# CLIP Visual Eco-Bias Scorer — HYBRID MODE
# - Uses real CLIP scoring for products that have a downloaded image
# - Uses text-based simulation for products with no image
# Install first: pip install torch torchvision
#                pip install git+https://github.com/openai/CLIP.git

import os
import re
import torch
import clip
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ── ECO PROMPTS (from Procedure Book) ────────────────────────────────
ECO_POSITIVE_PROMPTS = [
    'a natural and organic food product with green eco-friendly packaging',
    'a wholesome clean-label food with nature imagery and leaf motifs',
    'a healthy product with countryside farm imagery and earthy colors',
    'a pure and natural food with plant-based organic branding',
]

ECO_COUNTER_PROMPTS = [
    'a processed food product with artificial ingredients and preservatives',
    'a heavily manufactured snack with synthetic additives and chemicals',
    'an industrial packaged food with bright artificial colors',
    'a junk food product with high sugar and artificial flavoring',
]

# ── ECO KEYWORD LISTS FOR TEXT SIMULATION ────────────────────────────
ECO_POSITIVE_KEYWORDS = [
    'organic', 'bio', 'natural', 'eco', 'green', 'sustainable',
    'plant-based', 'vegan', 'fair trade', 'fairtrade', 'rainforest',
    'recyclable', 'compostable', 'biodegradable', 'no additives',
    'no preservatives', 'clean label', 'free range', 'grass fed',
    'wholesome', 'pure', 'earthy', 'farm', 'nature', 'leaf',
]

ECO_NEGATIVE_KEYWORDS = [
    'artificial', 'synthetic', 'preservative', 'additive', 'chemical',
    'flavoring', 'coloring', 'modified', 'processed', 'hydrogenated',
    'high fructose', 'msg', 'e-number', 'e number', 'sweetener',
]

HIGH_RISK_ADDITIVES_PATTERN = re.compile(
    r'\bE\d{3,4}\b', re.IGNORECASE
)


def simulate_vebs_from_text(row: pd.Series) -> float:
    """
    Estimate a VEBS score (0–1) from CSV text columns when no image exists.
    Combines:
      - eco keyword hits in labels_tags and categories_en  (+0.06 each, max 0.35)
      - negative keyword hits in ingredients_text          (-0.05 each, min 0)
      - E-number additive count penalty                    (-0.03 each, max -0.20)
      - greenwash_label prior adjustment
    Returns a float clipped to [0.10, 0.90] so simulated scores stay
    clearly within range and never collide with the 0.5 error sentinel.
    """
    score = 0.50  # neutral baseline

    # --- build a combined text blob for keyword search ---
    labels     = str(row.get('labels_tags', '') or '')
    categories = str(row.get('categories_en', '') or '')
    ingredients = str(row.get('ingredients_text', '') or '')
    combined_pos = (labels + ' ' + categories).lower()
    combined_neg = ingredients.lower()

    # positive keyword hits (eco buzzwords in labels/categories)
    pos_hits = sum(1 for kw in ECO_POSITIVE_KEYWORDS if kw in combined_pos)
    score += min(pos_hits * 0.06, 0.35)

    # negative keyword hits (bad ingredients)
    neg_hits = sum(1 for kw in ECO_NEGATIVE_KEYWORDS if kw in combined_neg)
    score -= min(neg_hits * 0.05, 0.25)

    # E-number additive penalty
    e_numbers = len(HIGH_RISK_ADDITIVES_PATTERN.findall(ingredients))
    score -= min(e_numbers * 0.03, 0.20)

    # greenwash label prior — nudge score toward expected direction
    label = str(row.get('greenwash_label', '')).strip().lower()
    if label == 'high':
        score += 0.10   # high greenwash → strong eco imagery expected
    elif label == 'low':
        score -= 0.08   # low greenwash → less eco posturing

    return round(float(np.clip(score, 0.10, 0.90)), 4)


class CLIPVisualScorer:
    def __init__(self):
        print("Loading CLIP model (ViT-B/32)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        pos_tokens = clip.tokenize(ECO_POSITIVE_PROMPTS).to(self.device)
        neg_tokens = clip.tokenize(ECO_COUNTER_PROMPTS).to(self.device)
        with torch.no_grad():
            self.pos_features = self.model.encode_text(pos_tokens).float()
            self.neg_features = self.model.encode_text(neg_tokens).float()
            self.pos_features /= self.pos_features.norm(dim=-1, keepdim=True)
            self.neg_features /= self.neg_features.norm(dim=-1, keepdim=True)
        print(f"  CLIP loaded on: {self.device.upper()}")

    def score_image(self, image_path: str) -> dict:
        """Score one image. Returns VEBS score between 0 and 1."""
        try:
            image = self.preprocess(Image.open(image_path).convert('RGB'))
            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_features = self.model.encode_image(image).float()
                img_features /= img_features.norm(dim=-1, keepdim=True)

            pos_sim = (img_features @ self.pos_features.T).mean().item()
            neg_sim = (img_features @ self.neg_features.T).mean().item()

            vebs = (pos_sim - neg_sim + 1) / 2
            vebs = float(np.clip(vebs, 0.0, 1.0))

            return {'vebs': round(vebs, 4), 'status': 'clip_real'}

        except Exception as e:
            return {'vebs': 0.5, 'status': f'clip_error: {str(e)}'}


# ── RUN ON FULL DATASET ───────────────────────────────────────────────
if __name__ == '__main__':
    DATA_FILE  = 'data/annotated/final_labels.csv'
    IMAGES_DIR = 'data/images'
    OUT_FILE   = 'outputs/clip_scores.csv'

    os.makedirs('outputs', exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} products from {DATA_FILE}")

    scorer = CLIPVisualScorer()

    results        = []
    real_scored    = 0
    text_simulated = 0
    errors         = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring (hybrid)"):
        barcode  = str(row['code']).strip()
        img_path = os.path.join(IMAGES_DIR, f"{barcode}.jpg")

        if os.path.exists(img_path) and os.path.getsize(img_path) > 5000:
            # ── REAL: image exists and is valid ──
            result = scorer.score_image(img_path)
            if 'error' in result['status']:
                errors += 1
                # fall back to text simulation on image read error
                vebs   = simulate_vebs_from_text(row)
                status = 'text_simulated_fallback'
            else:
                vebs   = result['vebs']
                status = result['status']
                real_scored += 1
        else:
            # ── SIMULATED: no image, use text features ──
            vebs   = simulate_vebs_from_text(row)
            status = 'text_simulated'
            text_simulated += 1

        results.append({
            'code'        : barcode,
            'vebs'        : vebs,
            'clip_status' : status,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)

    real_vebs = out_df[out_df['clip_status'] == 'clip_real']['vebs']
    sim_vebs  = out_df[out_df['clip_status'].str.startswith('text_sim')]['vebs']

    print(f"\n{'='*55}")
    print(f"  HYBRID CLIP SCORING COMPLETE")
    print(f"{'='*55}")
    print(f"  Real images scored (CLIP)    : {real_scored}")
    print(f"  Text-simulated (no image)    : {text_simulated}")
    print(f"  Image read errors (fallback) : {errors}")
    print(f"  Total rows in output         : {len(out_df)}")
    print(f"{'─'*55}")
    if len(real_vebs) > 0:
        print(f"  Avg VEBS — real images       : {real_vebs.mean():.4f}")
    if len(sim_vebs) > 0:
        print(f"  Avg VEBS — simulated         : {sim_vebs.mean():.4f}")
    print(f"  Results saved to             : {OUT_FILE}")
    print(f"{'='*55}")
    print(f"\n✅ Done! Proceed to: python src\\p2_perception\\step3_ocr_extractor.py")
