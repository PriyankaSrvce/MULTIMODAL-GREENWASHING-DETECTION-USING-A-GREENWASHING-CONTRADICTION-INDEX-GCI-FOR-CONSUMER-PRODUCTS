# PERSON 2 — GREENWASH GUARD SETUP & RUN GUIDE
## Everything you need to do, step by step

---

## WHAT YOU HAVE (already done for you)

All 4 scripts are already written. You just need to:
1. Set up the environment
2. Run ONE command

---

## STEP 1 — Copy your folder from Person 1

Your project folder should look like this:

```
greenwash_guard/
├── data/
│   ├── filtered/off_filtered.parquet   ← from Person 1
│   └── annotated/final_labels.csv      ← from Person 1
├── src/
│   ├── __init__.py
│   └── p2_perception/
│       ├── __init__.py
│       ├── download_images.py
│       ├── clip_scorer.py
│       ├── ocr_extractor.py
│       └── claim_detector.py
├── outputs/                            ← will be created automatically
├── run_person2.py                      ← YOUR MAIN SCRIPT
└── README_PERSON2.md
```

---

## STEP 2 — Install Tesseract (one time only)

### Windows:
1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the `.exe` installer
3. Install it (note the path, e.g. `C:\Program Files\Tesseract-OCR`)
4. Add that path to Windows PATH (search "Environment Variables" in Start menu)

### Verify Tesseract works:
Open Command Prompt and type:
```
tesseract --version
```
Should print version 5.x.x

---

## STEP 3 — Activate your virtual environment

Open Command Prompt in the `greenwash_guard/` folder:

```
venv\Scripts\activate
```

You should see `(venv)` at the start of the line.

---

## STEP 4 — Install extra packages (one time only)

```
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install transformers pytesseract Pillow opencv-python tqdm pandas
```

> **Note:** CLIP and DeBERTa models download automatically on first run.
> CLIP = ~300MB, DeBERTa = ~800MB. This is normal.

---

## STEP 5 — Run everything

From the `greenwash_guard/` folder:

```
python run_person2.py
```

That's it. This script will:
- ✅ Download all 500 product images
- ✅ Score each image with CLIP (VEBS score)
- ✅ Extract text from images with OCR
- ✅ Detect eco-claims with DeBERTa (ECS score)
- ✅ Save final output to `outputs/p2_perception_scores.csv`

---

## EXPECTED RUNTIME

| Step | Time |
|------|------|
| Download images (first time) | 10–20 mins |
| CLIP scoring (500 images, CPU) | 15–30 mins |
| OCR extraction (500 images) | 5–15 mins |
| Claim detection (500 products, CPU) | 30–60 mins |
| **Total** | **~1–2 hours** |

> Run it before going to sleep or before class — it runs unattended.

---

## OUTPUT

After running, share this file with Person 3:
```
outputs/p2_perception_scores.csv
```

This CSV has columns: `code`, `product_name`, `vebs`, `ecs`, `claim_count`, `ocr_text`, etc.

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `tesseract not found` | Set path in `ocr_extractor.py` line 13: uncomment and edit the `pytesseract.tesseract_cmd` line |
| `CLIP install fails` | Run `pip install torch torchvision` first, then CLIP |
| `Model download hangs` | Wait — it resumes from checkpoint. Run again if interrupted. |
| `Memory error` | Close other programs. The DeBERTa model needs ~4GB RAM. |
| `No module named clip` | Run: `pip install git+https://github.com/openai/CLIP.git` |

---

## IF SOMETHING FAILS MIDWAY

Each step saves its own output file:
- After Step 2: `outputs/clip_scores.csv`
- After Step 3: `outputs/ocr_results.csv`
- After Step 4: `outputs/claim_scores.csv`

You can re-run individual scripts:
```
python src/p2_perception/clip_scorer.py
python src/p2_perception/ocr_extractor.py
python src/p2_perception/claim_detector.py
```
