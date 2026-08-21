"""
typo_detective.py
Module 2 of the Cascade Verification System: Text / Typo Detection

Uses Tesseract OCR (via pytesseract) to read the text printed on a
product label or box, then compares it against a known-correct "master"
brand string using difflib.SequenceMatcher — a Ratcliff/Obershelp
similarity ratio that behaves like a normalized Levenshtein distance
(1.0 = identical strings, 0.0 = completely dissimilar).

Counterfeit packaging frequently contains subtle spelling mistakes
(e.g. "Nlke" instead of "Nike", "Adiddas" instead of "Adidas") that are
easy to miss visually but are reliably caught by string similarity.

macOS setup:
    brew install tesseract
    pip install pytesseract pillow opencv-python
"""

import re
from difflib import SequenceMatcher

import cv2
import pytesseract

SIMILARITY_THRESHOLD = 0.80  # 0..1, minimum ratio to be considered authentic text


#  Tesseract page-segmentation mode: 11 = "sparse text, no particular
#  order". Product photos usually mix a logo, shapes and a QR/barcode in
#  the same frame as the label text, which confuses the default
#  full-page layout analysis (psm 3); sparse-text mode is far more
#  reliable at pulling the brand string out of that kind of frame.
OCR_CONFIG = "--psm 11"


def _preprocess_for_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"typo_detective: could not read image at '{image_path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Global Otsu thresholding is used deliberately over a local adaptive
    # threshold: adaptive thresholding fragments large solid shapes
    # (logos, borders) into high-frequency noise that further confuses
    # Tesseract's layout analysis, whereas Otsu preserves clean glyph
    # edges for the text region.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _clean_text(raw_text):
    text = raw_text.upper()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text(image_path):
    """Run OCR on image_path and return cleaned, uppercase alphanumeric text."""
    processed = _preprocess_for_ocr(image_path)
    raw = pytesseract.image_to_string(processed, config=OCR_CONFIG)
    return _clean_text(raw)


def _best_substring_ratio(master, extracted):
    """
    Slide a window the size of the master string across the extracted OCR
    text and keep the best-aligned similarity. OCR output often contains
    extra surrounding text (sizes, dates, barcodes) beyond just the brand
    name, so comparing the whole strings directly under-scores genuine
    matches.
    """
    whole_ratio = SequenceMatcher(None, master, extracted).ratio()

    words = extracted.split()
    master_word_count = max(len(master.split()), 1)
    best = whole_ratio
    for i in range(len(words)):
        window = " ".join(words[i:i + master_word_count])
        r = SequenceMatcher(None, master, window).ratio()
        if r > best:
            best = r
    return best


def verify(master_brand_string, test_path, threshold=SIMILARITY_THRESHOLD, debug=False):
    """
    Compare OCR'd text of test_path with the ground-truth brand string.

    Parameters
    ----------
    master_brand_string : str, e.g. "NIKE AIR MAX AUTHENTIC"
    test_path            : str, path to the test image

    Returns
    -------
    dict: passed, extracted_text, best_match_ratio, score, message
    """
    master_clean = _clean_text(master_brand_string)
    extracted = extract_text(test_path)

    if not extracted:
        return {
            "passed": False,
            "extracted_text": "",
            "best_match_ratio": 0.0,
            "score": 0.0,
            "message": "OCR could not detect any legible text on the label.",
        }

    ratio = _best_substring_ratio(master_clean, extracted)
    passed = ratio >= threshold

    if debug:
        print(f"[typo_detective] Master : '{master_clean}'")
        print(f"[typo_detective] Extract: '{extracted}'")
        print(f"[typo_detective] Ratio  : {ratio:.3f}")

    return {
        "passed": passed,
        "extracted_text": extracted,
        "best_match_ratio": round(ratio, 3),
        "score": round(ratio, 3),
        "message": f"Text similarity {ratio:.2%} (threshold={threshold:.0%}).",
    }


def verify_gallery(brand_strings, test_path, threshold=SIMILARITY_THRESHOLD, debug=False):
    """
    One-vs-many variant of verify(): OCR the test image once, then check
    it against a list of acceptable brand strings and keep the best
    match. Suited to a checkpoint that authenticates several brands
    (e.g. ["NIKE", "PUMA"]) rather than a single fixed product label.

    Parameters
    ----------
    brand_strings : list[str] of acceptable ground-truth brand texts

    Returns
    -------
    dict: passed, extracted_text, best_match_ratio, best_brand, score, message
    """
    if not brand_strings:
        raise ValueError("typo_detective.verify_gallery: brand_strings must not be empty.")

    extracted = extract_text(test_path)

    if not extracted:
        return {
            "passed": False,
            "extracted_text": "",
            "best_match_ratio": 0.0,
            "best_brand": None,
            "score": 0.0,
            "message": "OCR could not detect any legible text on the label.",
        }

    best_ratio = -1.0
    best_brand = None
    for brand_string in brand_strings:
        ratio = _best_substring_ratio(_clean_text(brand_string), extracted)
        if ratio > best_ratio:
            best_ratio = ratio
            best_brand = brand_string

    passed = best_ratio >= threshold

    if debug:
        print(f"[typo_detective] Extract: '{extracted}'")
        print(f"[typo_detective] Best brand: '{best_brand}' ratio={best_ratio:.3f}")

    return {
        "passed": passed,
        "extracted_text": extracted,
        "best_match_ratio": round(best_ratio, 3),
        "best_brand": best_brand,
        "score": round(best_ratio, 3),
        "message": f"Best text similarity {best_ratio:.2%} vs '{best_brand}' (threshold={threshold:.0%}).",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python typo_detective.py '<MASTER BRAND STRING>' <test_image>")
        sys.exit(1)

    result = verify(sys.argv[1], sys.argv[2], debug=True)
    print(result)
