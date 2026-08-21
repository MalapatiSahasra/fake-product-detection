"""
sort_kaggle_dataset.py
One-off sorting utility (not one of the 4 core cascade modules).

Takes the extracted Kaggle "Fake Brand Dataset (Adidas, Nike, Gucci,
Puma, LV)" by prashanthgolla
(https://www.kaggle.com/datasets/prashanthgolla/fake-brand-data,
CC BY-NC 4.0) and sorts the requested brand subfolders
(<brand>_genuine/, <brand>_fake/) into this project's layout:

    dataset/gallery/<brand>_<index>.jpg          (genuine only, held out)
    dataset/seed/genuine/<brand>_<index>.jpg      (genuine, for evaluation)
    dataset/seed/counterfeit/<brand>_<index>.jpg  (counterfeit, for evaluation)

A random, seeded subset of the genuine images is reserved into
dataset/gallery/ - the one-vs-many reference gallery consumed by
logo_matcher.verify_gallery() / color_inspector.verify_gallery(). Those
images are deliberately EXCLUDED from dataset/seed/genuine so the
evaluation set never contains an image that is also sitting in its own
reference gallery: without this split, a genuine test image could match
itself, silently inflating accuracy (data leakage).

Every image is re-encoded to JPEG via OpenCV (dropping any alpha
channel) so downstream modules (which all call cv2.imread) see a
consistent format.

Run:
    python3 sort_kaggle_dataset.py --source <path to extracted train/ dir> --brands nike puma
"""

import argparse
import os
import random

import cv2

DEFAULT_BRANDS = ["nike", "puma"]
GALLERY_FRACTION = 0.08  # share of genuine images reserved as the held-out gallery
RANDOM_SEED = 42


def _list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))
    )


def _write(src_path, out_dir, out_name):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    cv2.imwrite(os.path.join(out_dir, out_name), img)
    return True


def sort_brands(source_dir, brands, genuine_out, counterfeit_out, gallery_out,
                 gallery_fraction=GALLERY_FRACTION):
    os.makedirs(genuine_out, exist_ok=True)
    os.makedirs(counterfeit_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    counts = {"genuine": 0, "gallery": 0, "counterfeit": 0}

    for brand in brands:
        # --- genuine: split into held-out gallery + evaluation seed pool ---
        src = os.path.join(source_dir, f"{brand}_genuine")
        if os.path.isdir(src):
            files = _list_images(src)
            shuffled = files[:]
            rng.shuffle(shuffled)
            n_gallery = max(1, round(len(shuffled) * gallery_fraction)) if shuffled else 0
            gallery_files = set(shuffled[:n_gallery])

            g_i = e_i = 0
            for fname in files:
                src_path = os.path.join(src, fname)
                if fname in gallery_files:
                    if _write(src_path, gallery_out, f"{brand}_{g_i:04d}.jpg"):
                        g_i += 1
                else:
                    if _write(src_path, genuine_out, f"{brand}_{e_i:04d}.jpg"):
                        e_i += 1
            counts["gallery"] += g_i
            counts["genuine"] += e_i
            print(f"[sort_kaggle_dataset] {brand} genuine: {e_i} -> {genuine_out}, "
                  f"{g_i} reserved -> {gallery_out}")
        else:
            print(f"[sort_kaggle_dataset] WARNING: '{src}' not found, skipping.")

        # --- counterfeit: no gallery split needed (gallery is genuine-only) ---
        src = os.path.join(source_dir, f"{brand}_fake")
        if os.path.isdir(src):
            files = _list_images(src)
            c_i = 0
            for fname in files:
                if _write(os.path.join(src, fname), counterfeit_out, f"{brand}_{c_i:04d}.jpg"):
                    c_i += 1
            counts["counterfeit"] += c_i
            print(f"[sort_kaggle_dataset] {brand} counterfeit: {c_i} images -> {counterfeit_out}")
        else:
            print(f"[sort_kaggle_dataset] WARNING: '{src}' not found, skipping.")

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort the Kaggle fake-brand-data dataset into seed + gallery folders.")
    parser.add_argument("--source", required=True, help="Path to the extracted dataset's 'train' directory")
    parser.add_argument("--brands", nargs="+", default=DEFAULT_BRANDS, help="Brand subfolder prefixes to include")
    parser.add_argument("--genuine_out", default="dataset/seed/genuine")
    parser.add_argument("--counterfeit_out", default="dataset/seed/counterfeit")
    parser.add_argument("--gallery_out", default="dataset/gallery")
    parser.add_argument("--gallery_fraction", type=float, default=GALLERY_FRACTION)
    args = parser.parse_args()

    result = sort_brands(
        args.source, args.brands, args.genuine_out, args.counterfeit_out, args.gallery_out,
        gallery_fraction=args.gallery_fraction,
    )
    print(f"[sort_kaggle_dataset] TOTAL genuine(eval)={result['genuine']}, "
          f"gallery(held-out)={result['gallery']}, counterfeit={result['counterfeit']}")
