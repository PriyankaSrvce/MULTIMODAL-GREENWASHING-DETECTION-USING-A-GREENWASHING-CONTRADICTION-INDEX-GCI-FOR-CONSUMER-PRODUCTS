# src/p2_perception/step4_claim_detector.py
# Eco-Claim Detector using DeBERTa-v3 NLI
# Detects greenwashing claims in product text
# NOTE: First run downloads ~800MB model automatically — needs internet

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ── ECO-CLAIM HYPOTHESES ──────────────────────────────────────────────
ECO_CLAIMS = [
    "this product is natural",
    "this product is organic",
    "this product contains no artificial ingredients",
    "this product is environmentally friendly",
    "this product is healthy",
    "this product contains no preservatives",
    "this product is pure and clean",
    "this product is sustainable",
]

class ClaimDetector:
    def __init__(self):
        print("Loading DeBERTa-v3 NLI model...")
        print("(First run downloads ~800MB — please wait)")
        self.classifier = pipeline(
            'zero-shot-classification',
            model='cross-encoder/nli-deberta-v3-small',
            device=0 if torch.cuda.is_available() else -1
        )
        print("Model loaded!")

    def detect(self, text: str) -> dict:
        """Detect eco-claims in text. Returns ECS score and detected claims."""
        if not text or len(text.strip()) < 5:
            return {'ecs': 0.0, 'claim_count': 0, 'claims_found': [], 'status': 'empty_text'}

        try:
            result = self.classifier(
                text[:512],        # limit text length
                ECO_CLAIMS,
                multi_label=True   # multiple claims can be true
            )

            # Claims detected where confidence > 0.5
            claims_found = [
                label for label, score
                in zip(result['labels'], result['scores'])
                if score > 0.5
            ]

            claim_count = len(claims_found)

            # ECS = Eco-Claim Score (0 to 1)
            ecs = min(claim_count / len(ECO_CLAIMS), 1.0)

            return {
                'ecs'         : round(ecs, 4),
                'claim_count' : claim_count,
                'claims_found': claims_found,
                'status'      : 'ok'
            }

        except Exception as e:
            return {'ecs': 0.0, 'claim_count': 0, 'claims_found': [], 'status': f'error: {str(e)}'}


# ── RUN ON FULL DATASET ───────────────────────────────────────────────
if __name__ == '__main__':
    DATA_FILE = 'data/annotated/final_labels.csv'
    OCR_FILE  = 'outputs/ocr_texts.csv'        # from step3
    OUT_FILE  = 'outputs/claim_scores.csv'

    os.makedirs('outputs', exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    # Merge OCR text if available
    if os.path.exists(OCR_FILE):
        ocr_df = pd.read_csv(OCR_FILE)[['code', 'ocr_text']]
        df = df.merge(ocr_df, on='code', how='left')
        df['ocr_text'] = df['ocr_text'].fillna('')
        print("OCR text merged into dataset")
    else:
        df['ocr_text'] = ''
        print("No OCR file found — using product_name + ingredients_text only")

    # Combine all text sources for richer claim detection
    df['full_text'] = (
        df['product_name'].fillna('') + ' ' +
        df['ingredients_text'].fillna('') + ' ' +
        df['ocr_text']
    ).str.strip()

    detector = ClaimDetector()
    results  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting claims"):
        barcode = str(row['code']).strip()
        text    = str(row['full_text'])

        result  = detector.detect(text)
        results.append({
            'code'        : barcode,
            'ecs'         : result['ecs'],
            'claim_count' : result['claim_count'],
            'claims_found': '|'.join(result['claims_found']),
            'claim_status': result['status']
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)

    ok = out_df[out_df['claim_status'] == 'ok']
    print(f"\n{'='*50}")
    print(f"  Products processed   : {len(ok)}")
    print(f"  Average ECS score    : {ok['ecs'].mean():.4f}")
    print(f"  Avg claims detected  : {ok['claim_count'].mean():.2f}")
    print(f"  Results saved to     : {OUT_FILE}")
    print(f"{'='*50}")
    print(f"\n✅ Done! Proceed to step5_merge_outputs.py")
