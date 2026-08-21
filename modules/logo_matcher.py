"""
logo_matcher.py
Module 1 of the Cascade Verification System: Logo / Feature Matching

Uses ORB (Oriented FAST and Rotated BRIEF) to detect keypoints and compute
binary descriptors, then matches a genuine reference image against the
test image with a Brute-Force Matcher using Hamming distance (the
correct distance metric for ORB's binary descriptors). A Lowe's ratio
test filters raw matches down to "good" matches, and a minimum
good-match threshold decides pass/fail.

Two matching modes are provided:
    verify(master_path, test_path)     - strict single-SKU mode: the test
                                          image must match ONE specific,
                                          known-genuine reference photo of
                                          the exact same physical product
                                          (e.g. a checkpoint verifying a
                                          unit against its known master).
    verify_gallery(gallery, test_path) - one-vs-many mode: the test image
                                          only needs to match the BEST of
                                          several genuine reference photos.
                                          Suited to brand-level datasets
                                          that span many different
                                          products/photo styles, where no
                                          single master image can
                                          represent "genuine" on its own.
                                          Use build_gallery() to precompute
                                          gallery features once and reuse
                                          them across many verify_gallery()
                                          calls instead of recomputing ORB
                                          features for the same references
                                          on every test image.

macOS setup:
    pip install opencv-contrib-python numpy
    (opencv-contrib-python is recommended for full feature-detector
    coverage; plain opencv-python also ships ORB.)
"""

import cv2

# --- Tunable parameters -----------------------------------------------
ORB_FEATURES = 1000     # max keypoints ORB will attempt to find per image
LOWE_RATIO = 0.75        # Lowe's ratio test threshold (lower = stricter)
MIN_GOOD_MATCHES = 15    # minimum "good" matches required to pass


def _load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"logo_matcher: could not read image at '{path}'")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _extract_features(path):
    img, gray = _load_grayscale(path)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp, des = orb.detectAndCompute(gray, None)
    return img, kp, des


def _match_descriptors(query_des, query_kp, train_des, train_kp):
    if query_des is None or train_des is None or len(query_kp) < 2 or len(train_kp) < 2:
        return []

    # crossCheck must be False to use knnMatch (k=2) for the ratio test.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(query_des, train_des, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)
    return good_matches


def verify(master_path, test_path, min_good_matches=MIN_GOOD_MATCHES, debug=False):
    """
    Compare the logo/features of test_path against a single master_path
    using ORB + BFMatcher. Intended for single-SKU verification, where
    master_path is a known-genuine reference photo of the exact product
    being checked.

    Returns
    -------
    dict:
        passed          : bool
        good_matches    : int
        total_keypoints : int (min keypoints found across the two images)
        score           : float in [0, 1] = good_matches / min_good_matches, capped at 1
        message         : str
    """
    master_img, master_kp, master_des = _extract_features(master_path)
    test_img, test_kp, test_des = _extract_features(test_path)

    good_matches = _match_descriptors(master_des, master_kp, test_des, test_kp)

    passed = len(good_matches) >= min_good_matches
    score = min(len(good_matches) / float(min_good_matches), 1.0)

    if debug:
        _show_debug(master_img, master_kp, test_img, test_kp, good_matches)

    return {
        "passed": passed,
        "good_matches": len(good_matches),
        "total_keypoints": min(len(master_kp), len(test_kp)),
        "score": round(score, 3),
        "message": f"{len(good_matches)} good matches found (threshold={min_good_matches}).",
    }


def build_gallery(gallery_paths):
    """
    Precompute ORB keypoints/descriptors for a list of genuine reference
    image paths, once, so verify_gallery() can be called many times
    (e.g. once per test image in a batch evaluation) without
    re-extracting features for the same reference images repeatedly.
    """
    gallery = []
    for path in gallery_paths:
        try:
            img, kp, des = _extract_features(path)
        except FileNotFoundError:
            continue
        if des is not None and len(kp) >= 2:
            gallery.append({"path": path, "img": img, "kp": kp, "des": des})
    return gallery


def verify_gallery(gallery, test_path, min_good_matches=MIN_GOOD_MATCHES, debug=False):
    """
    One-vs-many variant of verify(): compare test_path against every
    image in `gallery` and keep the best (highest good_matches) result.

    Parameters
    ----------
    gallery : list[str] of image paths, or a pre-built gallery from
              build_gallery() (preferred for repeated calls).

    Returns
    -------
    dict: passed, good_matches, best_reference, score, message
    """
    if not gallery:
        raise ValueError("logo_matcher.verify_gallery: gallery must not be empty.")
    if isinstance(gallery[0], str):
        gallery = build_gallery(gallery)

    test_img, test_kp, test_des = _extract_features(test_path)

    best_matches = 0
    best_ref = None
    for entry in gallery:
        good = _match_descriptors(entry["des"], entry["kp"], test_des, test_kp)
        if best_ref is None or len(good) > best_matches:
            best_matches = len(good)
            best_ref = entry["path"]

    passed = best_matches >= min_good_matches
    score = min(best_matches / float(min_good_matches), 1.0)

    if debug:
        print(f"[logo_matcher] best gallery match: {best_ref} ({best_matches} good matches)")

    return {
        "passed": passed,
        "good_matches": best_matches,
        "best_reference": best_ref,
        "score": round(score, 3),
        "message": f"Best gallery match: {best_matches} good matches vs '{best_ref}' (threshold={min_good_matches}).",
    }


def _show_debug(master_img, kp1, test_img, kp2, good_matches):
    match_img = cv2.drawMatches(
        master_img, kp1, test_img, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Logo Matcher - Good Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python logo_matcher.py <master_image> <test_image>")
        sys.exit(1)

    result = verify(sys.argv[1], sys.argv[2], debug=True)
    print(result)
