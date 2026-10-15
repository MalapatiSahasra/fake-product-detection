# Fake Product Detection using Image Matching

Master Comparator: a training-free, few-shot cascade for verifying whether a
photographed product is genuine or counterfeit, using a small gallery of genuine
reference photos (no neural network, no training). Four
independent checks vote; a photo passes if at most one disagrees.

**Live demo:** https://web-tau-indol-42.vercel.app

## How it works

| # | Module | Technique |
|---|---|---|
| 1 | `modules/logo_matcher.py` | ORB keypoints + Brute-Force Hamming matcher, Lowe's ratio test |
| 2 | `modules/color_inspector.py` | HSV Hue–Saturation histogram correlation (Value channel dropped for lighting invariance) |
| 3 | `modules/typo_detective.py` | Tesseract OCR + `difflib.SequenceMatcher` string similarity against known brand strings |
| 4 | `modules/qr_scanner.py` | `pyzbar` QR/barcode decode + serial-format validation |

`main.py` runs all four against a **gallery** of genuine reference images
(one-vs-many matching, not a single fixed master — real product photos
span too many angles/models/lighting conditions for one reference image to
represent "genuine") and returns a verdict.

## Measured accuracy

Evaluated end-to-end on 1,600 real product photos (800 genuine, 800
counterfeit) spanning **five brands — Nike, Puma, Adidas, Gucci, and LV**
— sourced from Kaggle and rebalanced with a held-out reference gallery to
avoid any test image ever matching itself:

| Metric | Value |
|---|---|
| Accuracy | 58.06% |
| Precision | 70.48% |
| Recall | 27.75% |
| F1-score | 0.398 |

QR/barcode checking was excluded from this run — only 1 of the 1,565 raw
source photos actually contained a scannable code, so it isn't a
meaningful signal on this kind of e-commerce/lifestyle photo dataset. See
`docs/dataset_statistics.md` for the full dataset breakdown and
methodology.

## Setup (macOS)

```bash
brew install tesseract zbar
pip install -r requirements.txt
```

## Usage

```bash
# Verify one photo against a gallery of genuine references
python3 main.py <gallery_dir> <test_image> --brand NIKE PUMA ADIDAS GUCCI LV

# Rebuild the seed dataset from the Kaggle "Fake Brand Dataset" download
python3 sort_kaggle_dataset.py --source <extracted_kaggle_dir>/train --brands nike puma adidas gucci lv

# Expand seed photos to a balanced evaluation set (800/class by default here)
python3 augment_data.py --target 800

# Run the full accuracy report
python3 evaluate_accuracy.py --brand NIKE PUMA ADIDAS GUCCI LV --no_qr
```

Don't have real product photos yet? `generate_seed_dataset.py` procedurally
renders a synthetic 10+10 seed set so the whole pipeline runs end-to-end
before you have real images.

## Project layout

```
modules/                 the 4 verification modules
main.py                  cascade aggregator
augment_data.py          rotation / brightness / noise augmentation to a balanced set
evaluate_accuracy.py     batch TP/TN/FP/FN + accuracy/precision/recall
sort_kaggle_dataset.py   sorts the Kaggle dataset into seed + held-out gallery folders
generate_seed_dataset.py synthetic seed-image generator (no real photos needed)
docs/dataset_statistics.md   dataset acquisition & characteristics writeup
docs/IEEE_paper.tex      IEEE conference-format paper (LaTeX source)
web/index.html           the live demo page (deployed to Vercel)
```

## Dataset

`dataset/` is gitignored — the images aren't committed. Source:
["Fake Brand Dataset (Adidas, Nike, Gucci, Puma, LV)"](https://www.kaggle.com/datasets/prashanthgolla/fake-brand-data)
by prashanthgolla on Kaggle, CC BY-NC 4.0. All five brand classes are
used. Regenerate the full dataset locally with `sort_kaggle_dataset.py`
followed by `augment_data.py --target 800`.
