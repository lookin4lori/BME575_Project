"""APTOS 2019 binary DR input pipeline (Section 1c).

Reads the existing split manifests and locally generated 224x224 RGB PNGs.
No split is created here. TensorFlow is imported only by ``build_tf_dataset``
so manifest checks, class weights, and the preview work without it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
BINARY_COLUMNS = (
    "binary_label",
    "binary_diagnosis",
    "diagnosis_binary",
    "label_binary",
    "dr_label",
)


@lru_cache(maxsize=1)
def _project_config():
    candidates = (
        Path.cwd() / "config.py",
        Path(__file__).resolve().parent / "config.py",
        Path(__file__).resolve().parent.parent / "config.py",
    )
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("section1c_project_config", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load project config: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def _config_value(name: str, fallback: Any) -> Any:
    config = _project_config()
    return getattr(config, name, fallback) if config is not None else fallback


def _integer_series(values: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} contains missing or nonnumeric labels")
    if not (numeric == np.floor(numeric)).all():
        raise ValueError(f"{column} must contain whole-number labels")
    return numeric.astype("int64")


def _binary_labels(frame: pd.DataFrame, label_column: str | None) -> pd.Series:
    if label_column is not None:
        if label_column not in frame:
            raise ValueError(f"Label column {label_column!r} is absent from the manifest")
        chosen = label_column
    else:
        chosen = next((name for name in BINARY_COLUMNS if name in frame), None)
        if chosen is None:
            chosen = "diagnosis" if "diagnosis" in frame else None
        if chosen is None:
            chosen = next((name for name in ("target", "label") if name in frame), None)
        if chosen is None:
            raise ValueError(
                "No binary label found. Supply label_column or a diagnosis column."
            )

    raw = _integer_series(frame[chosen], column=chosen)
    if chosen == "diagnosis":
        if not raw.between(0, 4).all():
            raise ValueError("diagnosis must contain APTOS grades 0 through 4")
        labels = (raw > 0).astype("int64")
    else:
        if not raw.isin((0, 1)).all():
            raise ValueError(f"{chosen} must be binary (0 or 1)")
        labels = raw

    if "diagnosis" in frame and chosen != "diagnosis":
        grades = _integer_series(frame["diagnosis"], column="diagnosis")
        if not grades.between(0, 4).all():
            raise ValueError("diagnosis must contain APTOS grades 0 through 4")
        if not labels.equals((grades > 0).astype("int64")):
            raise ValueError(f"{chosen} disagrees with diagnosis > 0")
    return labels


def read_manifest(
    csv_path: str | Path,
    image_dir: str | Path,
    *,
    label_column: str | None = None,
    id_column: str | None = None,
    image_extension: str | None = None,
    check_images: bool = True,
) -> pd.DataFrame:
    """Return manifest rows with ``binary_label`` and ``image_path`` columns.

    ``diagnosis`` grades 1-4 map to DR=1; grade 0 maps to DR=0. An explicit
    binary column is checked against ``diagnosis`` when both are available.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    id_column = id_column or str(_config_value("ID_COLUMN", "id_code"))
    extension = image_extension or str(_config_value("IMAGE_EXTENSION", ".png"))
    if not extension.startswith("."):
        extension = "." + extension
    if extension.lower() != ".png":
        raise ValueError("Section 1c expects processed PNG images")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype={id_column: "string"})
    if frame.empty:
        raise ValueError(f"Split manifest is empty: {csv_path}")
    if id_column not in frame:
        raise ValueError(f"ID column {id_column!r} is absent from {csv_path}")
    ids = frame[id_column]
    if ids.isna().any():
        raise ValueError(f"{id_column} contains missing IDs")
    ids = ids.str.strip()
    if (ids == "").any() or ids.str.contains(r"[/\\]", regex=True).any():
        raise ValueError(f"{id_column} contains an empty or invalid filename")
    if ids.duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate {id_column} values")

    filename_column = str(_config_value("FILENAME_COLUMN", "filename"))
    if filename_column in frame:
        expected_names = ids + extension
        given_names = frame[filename_column].astype("string").str.strip()
        if given_names.isna().any() or not given_names.equals(expected_names):
            raise ValueError(f"{filename_column} does not match {id_column} plus {extension}")

    result = frame.copy()
    result[id_column] = ids
    result["binary_label"] = _binary_labels(result, label_column).to_numpy()
    result["image_path"] = [str(image_dir / f"{image_id}{extension}") for image_id in ids]
    if check_images:
        missing = [path for path in result["image_path"] if not Path(path).is_file()]
        if missing:
            example = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} processed images are missing; first paths: {example}"
            )
    return result


def load_splits(
    *,
    split_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    label_column: str | None = None,
    check_images: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load committed train/val/test CSVs and reject ID overlap."""
    split_dir = Path(split_dir or _config_value("SPLIT_DIR", "data/splits"))
    image_dir = Path(
        image_dir or _config_value("PREPROCESSED_IMAGE_DIR", "data/processed/aptos2019_224")
    )
    id_column = str(_config_value("ID_COLUMN", "id_code"))
    splits = {
        name: read_manifest(
            split_dir / f"{name}.csv",
            image_dir,
            label_column=label_column,
            id_column=id_column,
            check_images=check_images,
        )
        for name in SPLITS
    }
    seen: set[str] = set()
    for name, frame in splits.items():
        split_column = str(_config_value("SPLIT_COLUMN", "split"))
        if split_column in frame:
            labels = frame[split_column].astype("string")
            if labels.isna().any() or labels.ne(name).any():
                raise ValueError(f"{name}.csv contains rows with a different split label")
        current = set(frame[id_column].tolist())
        overlap = seen & current
        if overlap:
            raise ValueError(f"{name}.csv overlaps an earlier split: {sorted(overlap)[:3]}")
        seen.update(current)
    return splits


def compute_class_weights(train_frame: pd.DataFrame) -> dict[int, float]:
    """Balanced weights n_train / (2 * n_class), from training labels only."""
    if "binary_label" not in train_frame:
        raise ValueError("Training frame must have a binary_label column")
    labels = _integer_series(train_frame["binary_label"], column="binary_label")
    if not labels.isin((0, 1)).all():
        raise ValueError("Training labels must be binary")
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError("Both classes must occur in the training split")
    return {int(c): float(len(labels) / (2 * counts[c])) for c in (0, 1)}


def build_tf_dataset(
    frame: pd.DataFrame,
    *,
    batch_size: int | None = None,
    training: bool = False,
    augment: bool | None = None,
    seed: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
):
    """Create a batched ``tf.data.Dataset`` for Keras DenseNet121.

    Geometric augmentation is applied only to training data, before the
    DenseNet ImageNet preprocessing function. The seed controls both shuffle
    and augmentation. Labels remain 0/1 float32.
    """
    batch_size = int(_config_value("DENSENET_BATCH_SIZE", 32)) if batch_size is None else batch_size
    seed = int(_config_value("RANDOM_SEED", 0)) if seed is None else seed
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if frame.empty:
        raise ValueError("Cannot build a dataset from an empty manifest")
    for column in ("image_path", "binary_label"):
        if column not in frame:
            raise ValueError(f"Manifest is missing {column}")
    if augment is None:
        augment = training
    if augment and not training:
        raise ValueError("Augmentation is permitted only for training data")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to build the tf.data pipeline") from exc

    height = int(image_height or _config_value("IMAGE_HEIGHT", 224))
    width = int(image_width or _config_value("IMAGE_WIDTH", 224))
    if height != width:
        raise ValueError("90-degree augmentation requires square processed images")
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["binary_label"].to_numpy(dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load_image(index, sample):
        path, label = sample
        image = tf.io.decode_png(tf.io.read_file(path), channels=0)
        image = tf.ensure_shape(image, (height, width, 3))
        image = tf.cast(image, tf.float32)
        if augment:
            keys = tf.random.split(
                tf.stack((tf.cast(seed, tf.int32), tf.cast(index, tf.int32))), num=3
            )
            image = tf.image.stateless_random_flip_left_right(image, keys[0])
            image = tf.image.stateless_random_flip_up_down(image, keys[1])
            turns = tf.random.stateless_uniform(
                (), keys[2], minval=0, maxval=4, dtype=tf.int32
            )
            image = tf.image.rot90(image, k=turns)
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    return (
        dataset.enumerate().map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def save_sample_batch_figure(
    train_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    columns: int = 4,
    seed: int | None = None,
    augmented: bool = True,
) -> Path:
    """Draw one reproducible training batch from real processed images.

    By default the grid uses a batch from ``build_tf_dataset`` and reverses
    DenseNet normalization for display. ``augmented=False`` draws raw PNGs,
    which is useful for inspecting preprocessing without TensorFlow.
    """
    from PIL import Image, ImageDraw, ImageFont

    batch_size = int(_config_value("DENSENET_BATCH_SIZE", 32)) if batch_size is None else batch_size
    seed = int(_config_value("RANDOM_SEED", 0)) if seed is None else seed
    if batch_size <= 0 or columns <= 0:
        raise ValueError("batch_size and columns must be positive")
    if train_frame.empty:
        raise ValueError("Training frame is empty")
    height = int(_config_value("IMAGE_HEIGHT", 224))
    width = int(_config_value("IMAGE_WIDTH", 224))
    images: list[Image.Image] = []
    captions: list[str] = []
    if augmented:
        batch_images, batch_labels = next(
            iter(build_tf_dataset(train_frame, batch_size=batch_size, training=True, seed=seed))
        )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for image, label in zip(batch_images.numpy(), batch_labels.numpy()):
            pixels = np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(pixels, mode="RGB"))
            captions.append("No DR (0)" if int(label) == 0 else "Any DR (1)")
    else:
        id_column = str(_config_value("ID_COLUMN", "id_code"))
        sample = train_frame.sample(n=min(batch_size, len(train_frame)), random_state=seed)
        for _, item in sample.iterrows():
            with Image.open(item["image_path"]) as source:
                if source.mode != "RGB" or source.size != (width, height):
                    raise ValueError(f"Expected {width}x{height} RGB PNG: {item['image_path']}")
                images.append(source.copy())
            captions.append(f"{item[id_column]}  DR={item['binary_label']}")
    rows = (len(images) + columns - 1) // columns
    padding, caption, header = 12, 30, 48
    grid = Image.new(
        "RGB",
        (
            columns * (width + padding) + padding,
            header + rows * (height + caption + padding) + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    try:
        caption_font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        caption_font = ImageFont.load_default()
        title_font = caption_font
    title = f"Training batch (n={len(images)}) - {'augmented' if augmented else 'processed RGB'}"
    draw.text((padding, 10), title, fill="black", font=title_font)
    for slot, (image, label) in enumerate(zip(images, captions)):
        x = padding + (slot % columns) * (width + padding)
        y = header + (slot // columns) * (height + caption + padding)
        grid.paste(image, (x, y))
        draw.text((x, y + height + 4), label, fill="black", font=caption_font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Section 1c class weights and batch figure")
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--raw-preview", action="store_true", help="Show processed PNGs without TensorFlow augmentation"
    )
    parser.add_argument(
        "--weights-only", action="store_true", help="Compute weights without needing processed images"
    )
    args = parser.parse_args()

    splits = load_splits(
        split_dir=args.split_dir,
        image_dir=args.image_dir,
        label_column=args.label_column,
        check_images=not args.weights_only,
    )
    train = splits["train"]
    weights = compute_class_weights(train)
    counts = train["binary_label"].value_counts().reindex([0, 1], fill_value=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "class_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "source": "train.csv only",
                "train_counts": {str(c): int(counts[c]) for c in (0, 1)},
                "class_weights": {str(c): weights[c] for c in (0, 1)},
                "formula": "n_train / (2 * n_class)",
                "split_counts": {
                    name: {
                        str(c): int(frame["binary_label"].eq(c).sum()) for c in (0, 1)
                    }
                    for name, frame in splits.items()
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Class weights: {weights_path}")
    if not args.weights_only:
        figure_path = save_sample_batch_figure(
            train,
            args.output_dir / "figures" / "sample_training_batch.png",
            batch_size=args.batch_size,
            augmented=not args.raw_preview,
        )
        print(f"Sample batch: {figure_path}")


if __name__ == "__main__":
    main()
