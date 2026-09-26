"""Image preprocessing for APTOS 2019 fundus images.

Follows Section 4.1 of Mohanty et al. (2023), Sensors 23(12):5726.
Pipeline: crop to retina -> resize to 224x224 -> Ben Graham enhancement
-> optional circular mask.
Run from the repo root with: python -m scripts.preprocess
"""

from pathlib import Path

import cv2
import numpy as np

# Settings. Move these into config.py after Loreta's branch is merged.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "aptos2019"
RAW_IMAGE_DIR = RAW_DIR / "train_images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aptos2019_224"

IMG_SIZE = 224          # from the paper
CROP_TOL = 4            # pixels darker than this count as black border
BLUR_SIGMA = 25         # Gaussian blur strength for Ben Graham
USE_CIRCLE_MASK = True  # trim the bright ring at the retina's edge
CIRCLE_SCALE = 0.95      # circle radius as a fraction of half the image width

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