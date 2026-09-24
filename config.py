"""Shared project configuration."""

"""Shared configuration for the BMME 575 DenseNet121 project."""

from pathlib import Path


# Project directories... project root finds location of file for all of us even if paths are diff 

PROJECT_ROOT = Path(__file__).resolve().parent # resolve means to eliminate parrent adn current dic to find file

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "aptos2019"
IMAGE_DIR = RAW_DATA_DIR / "train_images"
LABELS_CSV = RAW_DATA_DIR / "train.csv"

SPLIT_DIR = DATA_DIR / "splits"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "reports"
TEST_DIR = PROJECT_ROOT / "tests"


# ============================================================
# Dataset columns
# ============================================================

ID_COLUMN = "id_code"
ORIGINAL_LABEL_COLUMN = "diagnosis"
BINARY_LABEL_COLUMN = "binary_label"
FILENAME_COLUMN = "filename"
SPLIT_COLUMN = "split"
HASH_COLUMN = "sha256"

IMAGE_EXTENSION = ".png"


# ============================================================
# Expected APTOS 2019 dataset information
# ============================================================

EXPECTED_TOTAL_IMAGES = 3662

EXPECTED_CLASS_COUNTS = {
    0: 1805,
    1: 370,
    2: 999,
    3: 193,
    4: 295,
}

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

BINARY_CLASS_NAMES = {
    0: "No DR",
    1: "Any DR",
}


# ============================================================
# Reproducible splitting configuration
# ============================================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20

# The paper does not specify stratification.
# We stratify by the original five-grade diagnosis so that every
# split retains similar representation from all severity levels.
STRATIFY_COLUMN = ORIGINAL_LABEL_COLUMN


# ============================================================
# Future paper-aligned DenseNet121 configuration
# ============================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

DENSENET_BATCH_SIZE = 32
DENSENET_EPOCHS = 50
DENSENET_INITIAL_LEARNING_RATE = 0.01
DENSENET_MINIMUM_LEARNING_RATE = 0.00005