"""
color_inspector.py
Module 3 of the Cascade Verification System: Color Profile Matching

Converts images from BGR to HSV color space and compares 2D
Hue-Saturation histograms. The Value channel is deliberately excluded
from the histogram so the comparison is invariant to lighting/exposure
differences between the reference photo and the test photo. Histograms
are normalized to [0, 1] and compared with the Correlation method
(cv2.HISTCMP_CORREL), which returns 1.0 for identical distributions and
values near 0 (or negative) for dissimilar ones.

Two matching modes are provided, mirroring logo_matcher.py:
    verify(master_path, test_path)     - strict single-SKU mode against
                                          one known-genuine reference.
    verify_gallery(gallery, test_path) - one-vs-many mode: passes if the
                                          test image's color profile is
                                          close to the BEST-matching
                                          image in a gallery of genuine
                                          references. Use build_gallery()
                                          to precompute gallery histograms
                                          once and reuse them across many
                                          verify_gallery() calls.

macOS setup:
    pip install opencv-python numpy
"""

import cv2

H_BINS = 50
S_BINS = 60
CORRELATION_THRESHOLD = 0.70  # 0..1, higher = stricter color match required


def _compute_hs_histogram(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"color_inspector: could not read image at '{image_path}'")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # channels [0, 1] = Hue, Saturation only -> channel 2 (Value) is
    # intentionally excluded for illumination invariance.
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def verify(master_path, test_path, threshold=CORRELATION_THRESHOLD, debug=False):
    """
    Compare the Hue-Saturation histogram of test_path against a single
    master_path. Intended for single-SKU verification, where master_path
    is a known-genuine reference photo of the exact product being checked.

    Returns
    -------
    dict: passed, correlation, score, message
    """
    master_hist = _compute_hs_histogram(master_path)
    test_hist = _compute_hs_histogram(test_path)

    correlation = cv2.compareHist(master_hist, test_hist, cv2.HISTCMP_CORREL)
    # Correlation lies in [-1, 1]; clip negative values to 0 for a clean 0..1 score.
    score = max(correlation, 0.0)
    passed = correlation >= threshold

    if debug:
        print(f"[color_inspector] Correlation: {correlation:.4f} (threshold={threshold})")

    return {
        "passed": passed,
        "correlation": round(float(correlation), 4),
        "score": round(float(score), 4),
        "message": f"HS-histogram correlation = {correlation:.3f} (threshold={threshold}).",
    }


def build_gallery(gallery_paths):
    """
    Precompute Hue-Saturation histograms for a list of genuine reference
    image paths, once, so verify_gallery() can be called many times
    without recomputing histograms for the same reference images.
    """
    gallery = []
    for path in gallery_paths:
        try:
            hist = _compute_hs_histogram(path)
        except FileNotFoundError:
            continue
        gallery.append({"path": path, "hist": hist})
    return gallery


def verify_gallery(gallery, test_path, threshold=CORRELATION_THRESHOLD, debug=False):
    """
    One-vs-many variant of verify(): compare test_path's color profile
    against every image in `gallery` and keep the best (highest
    correlation) result.

    Parameters
    ----------
    gallery : list[str] of image paths, or a pre-built gallery from
              build_gallery() (preferred for repeated calls).

    Returns
    -------
    dict: passed, correlation, best_reference, score, message
    """
    if not gallery:
        raise ValueError("color_inspector.verify_gallery: gallery must not be empty.")
    if isinstance(gallery[0], str):
        gallery = build_gallery(gallery)

    test_hist = _compute_hs_histogram(test_path)

    best_corr = None
    best_ref = None
    for entry in gallery:
        corr = cv2.compareHist(entry["hist"], test_hist, cv2.HISTCMP_CORREL)
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_ref = entry["path"]

    score = max(best_corr, 0.0)
    passed = best_corr >= threshold

    if debug:
        print(f"[color_inspector] best gallery match: {best_ref} (corr={best_corr:.3f})")

    return {
        "passed": passed,
        "correlation": round(float(best_corr), 4),
        "best_reference": best_ref,
        "score": round(float(score), 4),
        "message": f"Best gallery correlation = {best_corr:.3f} vs '{best_ref}' (threshold={threshold}).",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python color_inspector.py <master_image> <test_image>")
        sys.exit(1)

    result = verify(sys.argv[1], sys.argv[2], debug=True)
    print(result)
