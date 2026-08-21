"""
augment_data.py
Data Augmentation Pipeline for the Fake Product Detection dataset.

Expands a small seed dataset (e.g. 10 genuine + 10 counterfeit photos,
placed in dataset/seed/genuine and dataset/seed/counterfeit) into a
balanced dataset of exactly 500 images per class (1000 total) by
applying randomized, physically-plausible camera-capture perturbations:

    1. Geometric rotation    : uniform random angle in [-15, +15] degrees
    2. Brightness shift      : random multiplicative scaling of the HSV
                                 Value channel (simulates indoor/outdoor/
                                 flash lighting conditions)
    3. Gaussian sensor noise : additive noise on the BGR image (simulates
                                 phone-camera sensor noise)

Each seed (physical product) photo is expanded into a deterministic,
reproducible set of derivative images via a fixed random seed, so the
mapping from seed image -> augmented images is stable across re-runs.
A per-category _manifest.csv records exactly which output file came
from which seed image and with which augmentation parameters, for
reproducibility in the accompanying paper.

macOS setup:
    pip install opencv-python numpy
"""

import argparse
import csv
import glob
import os
import random

import cv2
import numpy as np

TARGET_PER_CLASS = 500
ROTATION_RANGE_DEG = 15
BRIGHTNESS_RANGE = (0.6, 1.4)          # multiplicative factor on HSV-V channel
GAUSSIAN_NOISE_SIGMA_RANGE = (5, 20)   # std-dev of additive Gaussian noise
RANDOM_SEED = 42
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder):
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def shift_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def augment_once(img, rng):
    angle = rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG)
    brightness_factor = rng.uniform(*BRIGHTNESS_RANGE)
    sigma = rng.uniform(*GAUSSIAN_NOISE_SIGMA_RANGE)

    out = rotate_image(img, angle)
    out = shift_brightness(out, brightness_factor)
    out = add_gaussian_noise(out, sigma)

    meta = {
        "rotation_deg": round(angle, 2),
        "brightness_factor": round(brightness_factor, 2),
        "noise_sigma": round(sigma, 2),
    }
    return out, meta


def augment_category(seed_dir, output_dir, category_label, target_count=TARGET_PER_CLASS):
    os.makedirs(output_dir, exist_ok=True)
    seed_paths = _list_images(seed_dir)
    n_seeds = len(seed_paths)
    if n_seeds == 0:
        raise FileNotFoundError(f"No seed images found in '{seed_dir}'.")
    if target_count < n_seeds:
        raise ValueError("target_count must be >= number of seed images.")

    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    manifest_rows = []
    img_counter = 0

    # 1. Carry the original seed photos through unchanged - they are
    #    valid, zero-perturbation samples of the physical product.
    for idx, path in enumerate(seed_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        out_name = f"{category_label}_{img_counter:04d}_seed{idx:02d}_orig.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), img)
        manifest_rows.append([out_name, os.path.basename(path), "original", "", "", ""])
        img_counter += 1

    # 2. Distribute the remaining augmented images evenly across seeds so
    #    every physical product contributes a comparable number of samples.
    remaining = target_count - n_seeds
    base_per_seed, extra = divmod(remaining, n_seeds)

    for idx, path in enumerate(seed_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        n_augments = base_per_seed + (1 if idx < extra else 0)
        for a in range(n_augments):
            aug_img, meta = augment_once(img, rng)
            out_name = f"{category_label}_{img_counter:04d}_seed{idx:02d}_aug{a:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), aug_img)
            manifest_rows.append([
                out_name, os.path.basename(path), "augmented",
                meta["rotation_deg"], meta["brightness_factor"], meta["noise_sigma"],
            ])
            img_counter += 1

    _write_manifest(output_dir, manifest_rows)
    print(f"[augment_data] {category_label}: generated {img_counter} images "
          f"from {n_seeds} seed photos -> '{output_dir}'")
    return img_counter


def _write_manifest(output_dir, rows):
    manifest_path = os.path.join(output_dir, "_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "output_filename", "source_seed_image", "type",
            "rotation_deg", "brightness_factor", "noise_sigma",
        ])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a seed dataset up to a fixed count per class.")
    parser.add_argument("--genuine_seed", default="dataset/seed/genuine")
    parser.add_argument("--counterfeit_seed", default="dataset/seed/counterfeit")
    parser.add_argument("--genuine_out", default="dataset/genuine")
    parser.add_argument("--counterfeit_out", default="dataset/counterfeit")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS)
    args = parser.parse_args()

    augment_category(args.genuine_seed, args.genuine_out, "genuine", args.target)
    augment_category(args.counterfeit_seed, args.counterfeit_out, "counterfeit", args.target)
