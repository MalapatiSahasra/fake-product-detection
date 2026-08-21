"""
qr_scanner.py
Module 4 of the Cascade Verification System: QR / Barcode Validation

Uses pyzbar to locate and decode QR codes and barcodes embedded in
packaging, then validates the decoded string against an expected format
(e.g. a brand's serial-number scheme) and, optionally, an exact expected
value pulled from the master/reference product.

macOS setup:
    brew install zbar
    pip install pyzbar pillow opencv-python
"""

import re

import cv2
from pyzbar.pyzbar import ZBarSymbol, decode

# Default validation pattern - adjust to match your brand's real serial
# number / authentication code scheme.
# Example matched format: "NIKE-AX7291-2024"
DEFAULT_PATTERN = r"^[A-Z]{2,10}-[A-Z0-9]{4,12}-\d{4}$"


def _load_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"qr_scanner: could not read image at '{image_path}'")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def scan(image_path):
    """Decode all QR/barcodes found in the image. Returns a list of {type, data} dicts."""
    gray = _load_grayscale(image_path)
    decoded_objects = decode(
        gray, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.EAN13, ZBarSymbol.CODE128]
    )

    results = []
    for obj in decoded_objects:
        try:
            text = obj.data.decode("utf-8")
        except UnicodeDecodeError:
            text = obj.data.decode("latin-1", errors="ignore")
        results.append({"type": obj.type, "data": text})
    return results


def verify(test_path, pattern=DEFAULT_PATTERN, expected_value=None, debug=False):
    """
    Scan test_path for a QR/barcode and validate its content.

    Parameters
    ----------
    pattern        : regex the decoded string must fully match (format validity)
    expected_value : optional exact string (e.g. the master product's known
                      authentication code) for a stricter, exact-match check

    Returns
    -------
    dict: passed, decoded_count, decoded_values, score, message
    """
    decoded = scan(test_path)

    if debug:
        print(f"[qr_scanner] Decoded {len(decoded)} code(s) from '{test_path}': {decoded}")

    if not decoded:
        return {
            "passed": False,
            "decoded_count": 0,
            "decoded_values": [],
            "score": 0.0,
            "message": "No QR code or barcode detected in the image.",
        }

    values = [d["data"] for d in decoded]
    format_ok = any(re.fullmatch(pattern, v) for v in values)

    if expected_value is not None:
        exact_match = expected_value in values
        passed = format_ok and exact_match
        message = f"Format valid={format_ok}, exact match to expected code={exact_match}."
    else:
        passed = format_ok
        message = f"Format valid={format_ok} (pattern={pattern})."

    return {
        "passed": passed,
        "decoded_count": len(decoded),
        "decoded_values": values,
        "score": 1.0 if passed else 0.0,
        "message": message,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qr_scanner.py <test_image> [expected_code]")
        sys.exit(1)

    expected = sys.argv[2] if len(sys.argv) > 2 else None
    result = verify(sys.argv[1], expected_value=expected, debug=True)
    print(result)
