"""
main.py
Cascade Aggregator for the Fake Product Detection System.

Runs a test product image through 4 independent verification modules
(logo feature matching, color profile matching, OCR text/typo checking,
and QR/barcode validation) and combines their pass/fail outcomes into a
final authenticity decision using majority-vote cascade logic:

    score = number of modules passed (out of the modules that ran)
    Verdict = AUTHENTIC   if score >= (modules_run - 1)   [allow one miss]
              COUNTERFEIT  otherwise

QR/barcode validation only applies when the product's packaging actually
carries a scannable code - many e-commerce/lifestyle product photos
don't. Pass require_qr=False to drop it from the vote for such a data
source rather than forcing every image to fail a check it structurally
cannot pass; the "allow one miss" rule then applies to the remaining 3
modules (matches 3-of-4 behaviour exactly when QR is included).

The logo and color modules compare the test image against a GALLERY of
genuine reference images (one-vs-many) rather than a single fixed
master, since "genuine" for a real brand spans many different physical
products and photo styles that no single reference image can represent.
A test image passes those two checks if it is close enough to the BEST
match in the gallery. For precise single-SKU verification (checking one
specific product against one specific known-genuine unit), pass a
gallery containing exactly one reference image.

Set debug=False (default) to suppress all image popups / debug prints so
batch evaluation (see evaluate_accuracy.py) runs silently.
"""

from modules import color_inspector, logo_matcher, qr_scanner, typo_detective


def verify_product(
    genuine_gallery_paths,
    test_image_path,
    brand_strings,
    expected_qr_value=None,
    debug=False,
    logo_gallery=None,
    color_gallery=None,
    require_qr=True,
):
    """
    Run the full 4-module cascade against a single test image.

    Parameters
    ----------
    genuine_gallery_paths : list of genuine reference image paths, used
        directly by the logo/color modules if logo_gallery/color_gallery
        are not supplied.
    test_image_path        : path to the image being verified
    brand_strings            : ground-truth brand text (str), or a list of
        acceptable brand texts (e.g. ["NIKE", "PUMA"]) when checkpointing
        more than one brand - the best match wins
    expected_qr_value        : optional exact QR/barcode value for a stricter check
    debug                    : if True, show match visualizations / debug prints
    logo_gallery, color_gallery : optional precomputed galleries from
        logo_matcher.build_gallery() / color_inspector.build_gallery().
        Pass these when calling verify_product() many times (e.g. batch
        evaluation) so gallery features aren't recomputed on every call.

    Returns
    -------
    dict with per-module results, total score, and final verdict.
    """
    results = {}

    try:
        gallery = logo_gallery if logo_gallery is not None else genuine_gallery_paths
        results["logo"] = logo_matcher.verify_gallery(gallery, test_image_path, debug=debug)
    except Exception as e:
        results["logo"] = {"passed": False, "score": 0.0, "message": f"ERROR: {e}"}

    try:
        gallery = color_gallery if color_gallery is not None else genuine_gallery_paths
        results["color"] = color_inspector.verify_gallery(gallery, test_image_path, debug=debug)
    except Exception as e:
        results["color"] = {"passed": False, "score": 0.0, "message": f"ERROR: {e}"}

    try:
        brands = [brand_strings] if isinstance(brand_strings, str) else brand_strings
        results["typo"] = typo_detective.verify_gallery(brands, test_image_path, debug=debug)
    except Exception as e:
        results["typo"] = {"passed": False, "score": 0.0, "message": f"ERROR: {e}"}

    if require_qr:
        try:
            results["qr"] = qr_scanner.verify(test_image_path, expected_value=expected_qr_value, debug=debug)
        except Exception as e:
            results["qr"] = {"passed": False, "score": 0.0, "message": f"ERROR: {e}"}

    max_score = len(results)
    pass_threshold = max_score - 1  # allow at most one module to miss
    score = sum(1 for r in results.values() if r["passed"])
    is_authentic = score >= pass_threshold

    return {
        "modules": results,
        "score": score,
        "max_score": max_score,
        "pass_threshold": pass_threshold,
        "is_authentic": is_authentic,
        "verdict": "AUTHENTIC" if is_authentic else "COUNTERFEIT",
    }


def print_report(result, image_name=""):
    print("=" * 55)
    print(f" Cascade Verification Report {('- ' + image_name) if image_name else ''}")
    print("=" * 55)
    for name, r in result["modules"].items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {name.upper():<8} score={r.get('score', 0):<6} {r.get('message', '')}")
    print("-" * 55)
    print(f"  Total Score : {result['score']} / {result['max_score']}")
    print(f"  Verdict     : {result['verdict']}")
    print("=" * 55)


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import random

    parser = argparse.ArgumentParser(description="Fake Product Detection - Cascade Verifier")
    parser.add_argument("gallery_dir", help="Directory of genuine reference images")
    parser.add_argument("test_image", help="Path to the test image being verified")
    parser.add_argument("--brand", nargs="+", default=["MASTER BRAND NAME"], help="Ground-truth brand text(s) for the OCR check")
    parser.add_argument("--qr", default=None, help="Expected QR/barcode value, if any")
    parser.add_argument("--gallery_size", type=int, default=25, help="Cap on gallery images used (random sample)")
    parser.add_argument("--no_qr", action="store_true", help="Drop the QR/barcode check (for photos with no code)")
    parser.add_argument("--debug", action="store_true", help="Show debug windows/prints")
    args = parser.parse_args()

    gallery_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(args.gallery_dir, ext))
    )
    if not gallery_paths:
        raise SystemExit(f"No images found in gallery directory '{args.gallery_dir}'.")
    if len(gallery_paths) > args.gallery_size:
        gallery_paths = random.Random(42).sample(gallery_paths, args.gallery_size)

    result = verify_product(
        gallery_paths,
        args.test_image,
        args.brand,
        expected_qr_value=args.qr,
        debug=args.debug,
        require_qr=not args.no_qr,
    )
    print_report(result, image_name=args.test_image)
