"""
evaluate_accuracy.py
Automated batch evaluator for the Cascade Verification System.

Runs every image in dataset/genuine (ground-truth label = Authentic) and
dataset/counterfeit (ground-truth label = Counterfeit) through the
main.py cascade and reports True Positives, True Negatives, False
Positives, False Negatives, Overall Accuracy, Precision, and Recall.

The logo/color modules compare each test image against a GALLERY of
genuine reference images from dataset/gallery/ (one-vs-many matching -
see main.py). That gallery MUST be disjoint from dataset/genuine: if a
genuine test image were also sitting in its own reference gallery it
could trivially match itself, silently inflating accuracy. Gallery
features (ORB descriptors, HS histograms) are built once up front and
reused for every test image, rather than recomputed per call.

Convention (positive class = "Authentic"):
    TP : genuine image     correctly   classified as Authentic
    TN : counterfeit image correctly   classified as Counterfeit
    FP : counterfeit image incorrectly classified as Authentic
    FN : genuine image     incorrectly classified as Counterfeit
"""

import argparse
import glob
import os
import random
import time

from main import verify_product
from modules import color_inspector, logo_matcher

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder):
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def _progress(done, total, width=30):
    frac = done / total if total else 1
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r  Evaluating [{bar}] {done}/{total}", end="", flush=True)


def evaluate(genuine_dir, counterfeit_dir, gallery_dir, brand_string,
             expected_qr=None, gallery_size=25, limit=None, debug=False, require_qr=True):
    genuine_images = _list_images(genuine_dir)
    counterfeit_images = _list_images(counterfeit_dir)
    gallery_paths = _list_images(gallery_dir)

    if not gallery_paths:
        raise SystemExit(f"No gallery images found in '{gallery_dir}'.")
    overlap = set(genuine_images) & set(gallery_paths)
    if overlap:
        raise SystemExit(
            f"{len(overlap)} image(s) appear in both the evaluation set and the "
            f"gallery - this would leak genuine test images into their own "
            f"reference gallery. Keep dataset/gallery/ disjoint from {genuine_dir}."
        )
    if len(gallery_paths) > gallery_size:
        gallery_paths = random.Random(42).sample(gallery_paths, gallery_size)

    if limit:
        genuine_images = genuine_images[:limit]
        counterfeit_images = counterfeit_images[:limit]

    print(f"  Building reference gallery from {len(gallery_paths)} genuine image(s)...")
    logo_gallery = logo_matcher.build_gallery(gallery_paths)
    color_gallery = color_inspector.build_gallery(gallery_paths)

    tp = tn = fp = fn = 0
    errors = []
    total = len(genuine_images) + len(counterfeit_images)
    done = 0
    start = time.time()

    for path in genuine_images:
        done += 1
        try:
            result = verify_product(
                gallery_paths, path, brand_string, expected_qr, debug=debug,
                logo_gallery=logo_gallery, color_gallery=color_gallery,
                require_qr=require_qr,
            )
            if result["is_authentic"]:
                tp += 1
            else:
                fn += 1
        except Exception as e:
            errors.append((path, str(e)))
        _progress(done, total)

    for path in counterfeit_images:
        done += 1
        try:
            result = verify_product(
                gallery_paths, path, brand_string, expected_qr, debug=debug,
                logo_gallery=logo_gallery, color_gallery=color_gallery,
                require_qr=require_qr,
            )
            if result["is_authentic"]:
                fp += 1
            else:
                tn += 1
        except Exception as e:
            errors.append((path, str(e)))
        _progress(done, total)

    elapsed = time.time() - start
    print()  # newline after progress bar
    _report(tp, tn, fp, fn, errors, elapsed)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": errors}


def _report(tp, tn, fp, fn, errors, elapsed):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("=" * 55)
    print(" Cascade Verification System - Evaluation Report")
    print("=" * 55)
    print(f"  Total images evaluated : {total}")
    print(f"  Time elapsed           : {elapsed:.1f}s")
    print("-" * 55)
    print(f"  True Positives  (TP)   : {tp}")
    print(f"  True Negatives  (TN)   : {tn}")
    print(f"  False Positives (FP)   : {fp}")
    print(f"  False Negatives (FN)   : {fn}")
    print("-" * 55)
    print(f"  Accuracy   : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision  : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  Recall     : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  F1-Score   : {f1:.4f}")
    print("=" * 55)

    if errors:
        print(f"  Warning: {len(errors)} image(s) raised errors and were skipped.")
        for path, msg in errors[:10]:
            print(f"    - {path}: {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cascade accuracy over the dataset.")
    parser.add_argument("--genuine_dir", default="dataset/genuine")
    parser.add_argument("--counterfeit_dir", default="dataset/counterfeit")
    parser.add_argument("--gallery_dir", default="dataset/gallery", help="Held-out genuine reference images")
    parser.add_argument("--brand", nargs="+", default=["MASTER BRAND NAME"], help="Ground-truth brand text(s) for the OCR check")
    parser.add_argument("--qr", default=None, help="Expected QR/barcode value, if any")
    parser.add_argument("--gallery_size", type=int, default=25, help="Cap on gallery images used (random sample)")
    parser.add_argument("--no_qr", action="store_true", help="Drop the QR/barcode check (for photos with no code)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per class, for quick smoke tests")
    args = parser.parse_args()

    evaluate(
        args.genuine_dir, args.counterfeit_dir, args.gallery_dir,
        args.brand, args.qr, args.gallery_size, args.limit, debug=False,
        require_qr=not args.no_qr,
    )
