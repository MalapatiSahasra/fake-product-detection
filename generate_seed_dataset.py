"""
generate_seed_dataset.py
Synthetic seed-dataset generator (demo/bootstrap utility, not one of the
4 core cascade modules).

You will normally supply your own 10 genuine + 10 counterfeit product
photographs in dataset/seed/genuine and dataset/seed/counterfeit. This
script exists so the rest of the pipeline (augment_data.py,
evaluate_accuracy.py) can be exercised end-to-end with a real,
reproducible dataset before you have physical product photos on hand.

It procedurally renders "product box" images with a logo (shapes),
brand text, and a QR code using OpenCV drawing primitives:

  - Genuine seeds  : consistent brand identity (same shapes, same text,
                       same QR payload), with only minor natural
                       photographic jitter between units.
  - Counterfeit seeds: each sample deliberately corrupts one or more of
                       the four verifiable attributes (color, logo
                       shape, text, QR payload) so the generated dataset
                       exercises every module and includes some
                       "near-miss" fakes that break only a single check
                       - a realistic stress test of the 3-out-of-4
                       cascade threshold.

Run:
    python3 generate_seed_dataset.py
"""

import os

import cv2
import numpy as np

OUT_ROOT = "dataset/seed"
MASTER_BRAND_TEXT = "NIKE AIR MAX AUTHENTIC"
MASTER_QR_VALUE = "NIKE-AX7291-2024"
MASTER_COLOR = (230, 230, 230)  # light gray, BGR

IMG_SIZE = 500


def make_qr(data, size=140):
    encoder = cv2.QRCodeEncoder.create()
    qr = encoder.encode(data)
    qr = cv2.resize(qr, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def render_product(
    box_color,
    text,
    qr_value,
    seed,
    draw_logo=True,
    logo_style="standard",
    draw_qr=True,
    text_font_scale=0.85,
):
    rng = np.random.RandomState(seed)
    img = np.full((IMG_SIZE, IMG_SIZE, 3), box_color, dtype=np.uint8)

    # texture confined to a top strip, away from text/logo/QR, so it
    # only contributes ORB keypoints without corrupting OCR/QR reads
    for _ in range(15):
        pt1 = (rng.randint(0, IMG_SIZE), rng.randint(0, 90))
        pt2 = (rng.randint(0, IMG_SIZE), rng.randint(0, 90))
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.line(img, pt1, pt2, color, 2)

    if draw_logo:
        if logo_style == "standard":
            cv2.rectangle(img, (60, 100), (200, 220), (0, 0, 0), 3)
            cv2.circle(img, (350, 160), 55, (0, 0, 0), 3)
        elif logo_style == "swapped":
            # counterfeit logo: shapes transposed/resized - a common
            # real-world counterfeit tell (wrong swoosh geometry, etc.)
            cv2.circle(img, (130, 160), 55, (0, 0, 0), 3)
            cv2.rectangle(img, (300, 100), (410, 190), (0, 0, 0), 3)
        elif logo_style == "none":
            pass

    if text:
        cv2.putText(
            img, text, (20, 290), cv2.FONT_HERSHEY_DUPLEX,
            text_font_scale, (0, 0, 0), 2, cv2.LINE_AA,
        )

    if draw_qr and qr_value:
        qr = make_qr(qr_value, size=140)
        img[340:480, 340:480] = qr

    return img


def jitter(img, seed, rotation=4, brightness=(0.92, 1.08), noise_sigma=4):
    rng = np.random.RandomState(seed + 1000)
    h, w = img.shape[:2]
    angle = rng.uniform(-rotation, rotation)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(*brightness), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    noise = rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def build_genuine_seeds(out_dir, n=10):
    os.makedirs(out_dir, exist_ok=True)
    base = render_product(MASTER_COLOR, MASTER_BRAND_TEXT, MASTER_QR_VALUE, seed=0)
    cv2.imwrite(os.path.join(out_dir, "genuine_seed00_master.jpg"), base)

    for i in range(1, n):
        img = jitter(base, seed=i)
        cv2.imwrite(os.path.join(out_dir, f"genuine_seed{i:02d}.jpg"), img)

    print(f"[generate_seed_dataset] wrote {n} genuine seeds -> {out_dir}")
    return base


# Each entry: (label, box_color, text, qr_value, logo_style)
# Deliberately spans "obvious fake" (fails 3-4 checks) down to
# "near-miss fake" (fails exactly 1 check) so the evaluator report
# reflects a realistic range of counterfeit quality.
COUNTERFEIT_RECIPES = [
    # obvious fakes - fail nearly everything
    ("wrong_color_text_qr", (60, 60, 200), "NLKE AIR MAX AUTHENTC", "FAKE-0000-0000", "swapped"),
    ("wrong_logo_text_qr", (60, 60, 200), "NIKEE AIR MAXX", "BAD-1111-1111", "swapped"),
    ("blank_label", (40, 120, 40), "", "", "none"),
    ("wrong_everything", (10, 200, 200), "AIRMAX FAKE STORE", "XYZ-9999-0000", "swapped"),
    # moderate fakes - fail 2 of 4 checks
    ("wrong_color_qr", (60, 60, 200), MASTER_BRAND_TEXT, "FAKE-0000-0000", "standard"),
    ("wrong_logo_text", MASTER_COLOR, "NIKEE AIR MAXX AUTHENTC", MASTER_QR_VALUE, "swapped"),
    ("wrong_text_qr", MASTER_COLOR, "NIK3 AIR MAKS AUTH3NTIC", "COUNTERFEIT-000", "standard"),
    # near-miss fakes - fail exactly 1 check (stress-tests the 3/4 threshold)
    ("near_miss_color_only", (140, 190, 235), MASTER_BRAND_TEXT, MASTER_QR_VALUE, "standard"),
    ("near_miss_qr_only", MASTER_COLOR, MASTER_BRAND_TEXT, "NIKE-AX7291-2099", "standard"),
    ("near_miss_text_only", MASTER_COLOR, "NIKE AIR MAX AUTHENTIQ", MASTER_QR_VALUE, "standard"),
]


def build_counterfeit_seeds(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (label, color, text, qr_value, logo_style) in enumerate(COUNTERFEIT_RECIPES):
        img = render_product(
            color, text, qr_value, seed=100 + i,
            draw_logo=(logo_style != "none"), logo_style=logo_style,
            draw_qr=bool(qr_value),
        )
        img = jitter(img, seed=100 + i, rotation=5, noise_sigma=5)
        out_name = f"counterfeit_seed{i:02d}_{label}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), img)

    print(f"[generate_seed_dataset] wrote {len(COUNTERFEIT_RECIPES)} counterfeit seeds -> {out_dir}")


if __name__ == "__main__":
    genuine_out = os.path.join(OUT_ROOT, "genuine")
    counterfeit_out = os.path.join(OUT_ROOT, "counterfeit")

    master = build_genuine_seeds(genuine_out, n=10)
    cv2.imwrite("dataset/master_reference.jpg", master)
    build_counterfeit_seeds(counterfeit_out)

    print(f"[generate_seed_dataset] master reference -> dataset/master_reference.jpg")
    print(f"[generate_seed_dataset] master brand string -> '{MASTER_BRAND_TEXT}'")
    print(f"[generate_seed_dataset] master QR value -> '{MASTER_QR_VALUE}'")
