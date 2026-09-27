"""Image preprocessing for APTOS 2019 fundus images.

Follows Section 4.1 of Mohanty et al. (2023), Sensors 23(12):5726.
Pipeline: crop to retina -> resize to 224x224 -> Ben Graham enhancement
-> optional circular mask.
Run from the repo root with: python -m scripts.preprocess
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config

# All settings live in config.py. Short local names are kept so the functions
# below, the notebook, and the tests keep working without changes.
RAW_DIR = config.RAW_DATA_DIR
RAW_IMAGE_DIR = config.IMAGE_DIR
PROCESSED_DIR = config.PREPROCESSED_IMAGE_DIR
ALL_SPLITS_CSV = config.SPLIT_DIR / "all_splits.csv"   # Loreta's manifest of all images

IMG_SIZE = config.IMAGE_HEIGHT   # 224, from the paper; height and width are equal
CROP_TOL = config.CROP_TOLERANCE
BLUR_SIGMA = config.BEN_GRAHAM_SIGMA
USE_CIRCLE_MASK = config.USE_CIRCLE_MASK
CIRCLE_SCALE = config.CIRCLE_SCALE

def load_image(path):
    """Read an image from disk and return it as an RGB array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_to_retina(img, tol=CROP_TOL):
    """Crop to the retina, using the largest bright region so small artifacts don't skew the crop."""
    # Convert to grayscale and mark pixels brighter than the tolerance.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)

    # Find the outlines of every separate bright region.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If nothing is bright enough, return the image unchanged instead of crashing.
    if not contours:
        return img

    # Keep the largest region (the retina) and crop to its bounding box.
    retina = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(retina)
    return img[y:y + h, x:x + w]

def resize_image(img, size=IMG_SIZE):
    """Resize the image to size x size pixels, as specified in the paper."""
    # INTER_AREA averages neighboring pixels, which gives the cleanest result when shrinking.
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def ben_graham(img, sigma=BLUR_SIGMA):
    """Subtract a blurred copy to even out lighting and enhance detail (Ben Graham method)."""
    # Heavily blur the image to estimate the local background color and brightness.
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Compute 4*image - 4*blurred + 128: removes the background, boosts detail, centers on mid-gray.
    return cv2.addWeighted(img, 4, blurred, -4, 128)

def apply_circular_mask(img, scale=CIRCLE_SCALE):
    """Replace everything outside a centered circle with mid-gray (128) to remove edge artifacts."""
    h, w = img.shape[:2]

    # Start with an all-black mask the same height and width as the image.
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a filled white circle in the center; its radius is a fraction of half the image size.
    radius = int(scale * min(h, w) / 2)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)

    # Copy the image and set every pixel outside the circle to mid-gray.
    out = img.copy()
    out[mask == 0] = 128
    return out

def preprocess_image(path):
    """Run the full preprocessing pipeline on one image file and return a 224x224 RGB image."""
    img = load_image(path)
    img = crop_to_retina(img)      # remove black border so the retina fills the frame
    img = resize_image(img)        # 224x224, as in the paper
    img = ben_graham(img)          # even out lighting, enhance vessels and lesions
    if USE_CIRCLE_MASK:
        img = apply_circular_mask(img)  # remove bright edge artifact
    return img

def save_image(img, path):
    """Save an RGB image as a PNG. OpenCV writes in BGR order, so convert first."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Could not write image: {path}")


def process_all(manifest=ALL_SPLITS_CSV, out_dir=PROCESSED_DIR, overwrite=False):
    """Preprocess every image listed in the split manifest and save it to out_dir.

    Returns a dict of counts (processed, skipped, failed) and a list of failed image IDs.
    """
    ids = pd.read_csv(manifest)[config.ID_COLUMN]
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    failed_ids = []
    total = len(ids)

    for i, img_id in enumerate(ids, start=1):
        out_path = out_dir / f"{img_id}{config.IMAGE_EXTENSION}"

        # Skip images that were already processed, unless asked to redo them.
        if out_path.exists() and not overwrite:
            counts["skipped"] += 1
        else:
            try:
                img = preprocess_image(RAW_IMAGE_DIR / f"{img_id}{config.IMAGE_EXTENSION}")
                save_image(img, out_path)
                counts["processed"] += 1
            except Exception as err:
                # Record the failure and keep going instead of stopping the whole run.
                counts["failed"] += 1
                failed_ids.append(img_id)
                print(f"  FAILED {img_id}: {err}")

        if i % 200 == 0 or i == total:
            print(f"  {i}/{total} images done")

    return counts, failed_ids


def main():
    """Command-line entry point: python -m scripts.preprocess [--overwrite]"""
    parser = argparse.ArgumentParser(description="Preprocess APTOS 2019 fundus images.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess images even if output files already exist.")
    args = parser.parse_args()

    print("Preprocessing settings:")
    print(f"  size={IMG_SIZE}  crop_tol={CROP_TOL}  sigma={BLUR_SIGMA}  "
          f"mask={USE_CIRCLE_MASK}  mask_scale={CIRCLE_SCALE}")
    print(f"  input:  {RAW_IMAGE_DIR}")
    print(f"  output: {PROCESSED_DIR}\n")

    counts, failed_ids = process_all(overwrite=args.overwrite)

    print("\nSummary:")
    print(f"  processed: {counts['processed']}")
    print(f"  skipped (already existed): {counts['skipped']}")
    print(f"  failed: {counts['failed']}")
    if failed_ids:
        print(f"  failed IDs: {failed_ids}")


if __name__ == "__main__":
    main()