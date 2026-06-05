# src/p2_perception/step3_ocr_extractor.py
# OCR Text Extractor using Tesseract
# Extracts visible text from product images
#
# INSTALL FIRST:
#   1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
#   2. Install it (default path: C:\Program Files\Tesseract-OCR\)
#   3. pip install pytesseract pillow

import os
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
from tqdm import tqdm

# ── WINDOWS PATH — set your Tesseract install path ───────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRExtractor:
    def __init__(self):
        # Verify Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract found and ready!")
        except Exception:
            print("ERROR: Tesseract not found!")
            print("Install from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("Then set the path at the top of this file.")
            raise

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """Enhance image for better OCR accuracy."""
        img = img.convert('L')                          # grayscale
        img = img.filter(ImageFilter.SHARPEN)           # sharpen
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)                     # boost contrast
        return img

    def extract(self, image_path: str) -> dict:
        """Extract text from one image."""
        try:
            img = Image.open(image_path).convert('RGB')
            processed = self.preprocess_image(img)

            # Extract text
            raw_text = pytesseract.image_to_string(
                processed,
                config='--psm 3 --oem 3'   # auto page segmentation, LSTM engine
            )
            clean_text = ' '.join(raw_text.split())  # remove extra whitespace

            return {
                'text'       : clean_text,
                'char_count' : len(clean_text),
                'status'     : 'ok'
            }

        except Exception as e:
            return {'text': '', 'char_count': 0, 'status': f'error: {str(e)}'}


# ── RUN ON FULL DATASET ───────────────────────────────────────────────
if __name__ == '__main__':
    DATA_FILE  = 'data/annotated/final_labels.csv'
    IMAGES_DIR = 'data/images'
    OUT_FILE   = 'outputs/ocr_texts.csv'

    os.makedirs('outputs', exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    extractor = OCRExtractor()

    results = []
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text"):
        barcode  = str(row['code']).strip()
        img_path = os.path.join(IMAGES_DIR, f"{barcode}.jpg")

        if not os.path.exists(img_path):
            missing += 1
            results.append({'code': barcode, 'ocr_text': '', 'char_count': 0, 'ocr_status': 'no_image'})
            continue

        result = extractor.extract(img_path)
        results.append({
            'code'       : barcode,
            'ocr_text'   : result['text'],
            'char_count' : result['char_count'],
            'ocr_status' : result['status']
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)

    ok = out_df[out_df['ocr_status'] == 'ok']
    print(f"\n{'='*50}")
    print(f"  Images processed     : {len(ok)}")
    print(f"  Images missing       : {missing}")
    print(f"  Avg chars extracted  : {ok['char_count'].mean():.1f}")
    print(f"  Results saved to     : {OUT_FILE}")
    print(f"{'='*50}")
    print(f"\n✅ Done! Proceed to step4_claim_detector.py")
