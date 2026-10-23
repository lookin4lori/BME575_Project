"""Section 3a: train the frozen DenseNet121 binary DR classifier.

The paper specifies Adam, an initial learning rate of 0.01, a minimum of
0.00005, batches of 32, 50 epochs, and a pretrained feature detector followed
by a softmax classifier. It does not define the learning-rate schedule, loss,
class weights, or augmentation. Those implementation choices are recorded in
each run's metadata; the existing 70/10/20 team split is preserved.

By default the frozen backbone's pooled features are computed once. With no
augmentation or dropout this gives the same trainable head and gradients as
recomputing a frozen backbone every epoch, and makes CPU training practical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from scripts.classifier import build_classifier, compile_classifier
from scripts.dataset import build_tf_dataset, compute_class_weights, load_splits


class LearningRateToLogs(tf.keras.callbacks.Callback):
    """Add the rate used for each epoch to the CSV history."""

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["learning_rate"] = self.epoch_learning_rate


def validate_training_settings(
    *,
    epochs: int,
    batch_size: int,
    initial_learning_rate: float,
    min_learning_rate: float,
    reduce_lr_factor: float,
    reduce_lr_patience: int,
    augment: bool,
    cache_features: bool,
) -> None:
    """Fail before touching data when requested settings are inconsistent."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not all(
        math.isfinite(rate) and rate > 0
        for rate in (initial_learning_rate, min_learning_rate)
    ):
        raise ValueError("learning rates must be finite positive numbers")
    if min_learning_rate > initial_learning_rate:
        raise ValueError("min_learning_rate cannot exceed initial_learning_rate")
    if not math.isfinite(reduce_lr_factor) or not 0 < reduce_lr_factor < 1:
        raise ValueError("reduce_lr_factor must be between 0 and 1")
    if reduce_lr_patience < 1:
        raise ValueError("reduce_lr_patience must be positive")
    if augment and cache_features:
        raise ValueError("feature caching cannot preserve per-epoch augmentation")


def make_callbacks(
    run_dir: str | Path,
    *,
    min_learning_rate: float,
    reduce_lr_factor: float,
    reduce_lr_patience: int,
) -> list[tf.keras.callbacks.Callback]:
    """Log every epoch, save the best weights, and reduce LR on val plateau.

    The paper gives a minimum learning rate but does not describe a scheduler.
    ReduceLROnPlateau(factor=0.5, patience=3) is this project's explicit choice.
    There is no early stopping because the requested baseline has 50 epochs.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        LearningRateToLogs(),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_trainable.weights.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def extract_pooled_features(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frozen GAP features in manifest order; do not change labels."""
    backbone = model.get_layer(config.BACKBONE_NAME)
    if backbone.trainable:
        raise ValueError("cached features require a frozen backbone")
    extractor = tf.keras.Model(
        model.input, model.get_layer("global_average_pooling").output
    )
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, raw_labels in dataset:
        features = np.asarray(extractor(images, training=False), dtype=np.float32)
        labels = np.asarray(raw_labels, dtype=np.float32).reshape(-1)
        if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError("training labels must be 0 or 1")
        feature_batches.append(features)
        label_batches.append(labels.astype(np.int32))
    if not feature_batches:
        raise ValueError("cannot extract features from an empty dataset")
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def build_cached_head(
    full_model: tf.keras.Model, *, learning_rate: float
) -> tf.keras.Model:
    """Compile a head sharing the full model's Dense layer weights."""
    if full_model.get_layer(config.BACKBONE_NAME).trainable:
        raise ValueError("cached head requires a frozen backbone")
    pool = full_model.get_layer("global_average_pooling")
    feature_input = tf.keras.Input(shape=(pool.output.shape[-1],), name="cached_features")
    output = feature_input
    for layer in full_model.layers[full_model.layers.index(pool) + 1 :]:
        output = layer(output)
    head = tf.keras.Model(feature_input, output, name="cached_dr_head")
    return compile_classifier(head, learning_rate=learning_rate)


def _json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_sha256(name: str) -> str:
    return hashlib.sha256((config.SPLIT_DIR / f"{name}.csv").read_bytes()).hexdigest()


def _binary_counts(frame) -> dict[str, int]:
    return {str(cls): int(frame["binary_label"].eq(cls).sum()) for cls in (0, 1)}


def _label_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        lambda images, labels: (images, tf.cast(labels, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def train(args: argparse.Namespace) -> Path:
    """Run training and return the run directory; test images are never fitted."""
    validate_training_settings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_learning_rate=args.initial_learning_rate,
        min_learning_rate=args.min_learning_rate,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        augment=args.augment,
        cache_features=args.cache_features,
    )
    for name in ("train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "status": "started",
        "paper": "Mohanty et al., Sensors 2023, 23, 5726, Sections 4.3-4.4",
        "paper_settings": {
            "architecture": "frozen pretrained DenseNet121 + global average pooling + binary softmax",
            "optimizer": "Adam",
            "initial_learning_rate": 0.01,
            "minimum_learning_rate": 0.00005,
            "batch_size": 32,
            "epochs": 50,
        },
        "project_choices": {
            "split": "existing stratified 70/10/20 train/val/test manifests",
            "loss": config.CLASSIFIER_LOSS,
            "class_weights": "n_train / (2 * n_class), from fit training rows only",
            "lr_callback": "ReduceLROnPlateau(val_loss, factor, patience, min_lr)",
            "augmentation": bool(args.augment),
            "dropout": 0.0,
            "feature_cache": bool(args.cache_features),
            "pretrained_weights": args.weights,
        },
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "initial_learning_rate": args.initial_learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "reduce_lr_factor": args.reduce_lr_factor,
            "reduce_lr_patience": args.reduce_lr_patience,
            "seed": config.RANDOM_SEED,
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
        },
        "tensorflow_version": tf.__version__,
        "manifest_sha256": {name: _manifest_sha256(name) for name in ("train", "val", "test")},
    }
    _json_write(run_dir / "metadata.json", metadata)

    try:
        splits = load_splits(check_images=True)
        train_frame = splits["train"]
        val_frame = splits["val"]
        if args.train_limit is not None:
            if args.train_limit < 2:
                raise ValueError("train_limit must allow both classes")
            if args.train_limit < len(train_frame):
                counts = train_frame["binary_label"].value_counts()
                no_dr = round(args.train_limit * counts[0] / len(train_frame))
                no_dr = max(1, min(no_dr, int(counts[0]), args.train_limit - 1))
                any_dr = min(args.train_limit - no_dr, int(counts[1]))
                no_dr = min(args.train_limit - any_dr, int(counts[0]))
                train_frame = pd.concat(
                    [
                        train_frame.loc[train_frame["binary_label"].eq(cls)].sample(
                            n=size, random_state=config.RANDOM_SEED
                        )
                        for cls, size in ((0, no_dr), (1, any_dr))
                    ],
                    ignore_index=True,
                )
        if args.val_limit is not None:
            val_frame = val_frame.sample(
                n=min(args.val_limit, len(val_frame)), random_state=config.RANDOM_SEED
            )
        weights = compute_class_weights(train_frame)
        metadata["split_counts"] = {
            name: {"size": len(frame), "binary": _binary_counts(frame)}
            for name, frame in splits.items()
        }
        metadata["fitted_train_counts"] = _binary_counts(train_frame)
        metadata["fitted_val_counts"] = _binary_counts(val_frame)
        metadata["class_weights"] = {str(cls): weights[cls] for cls in (0, 1)}
        metadata["status"] = "loading_model"
        _json_write(run_dir / "metadata.json", metadata)

        model = build_classifier(
            weights=None if args.weights == "none" else args.weights,
            unfreeze_last_n_blocks=0,
            dropout_rate=0.0,
            # Cached-head training owns the optimizer. Leave the full image
            # model uncompiled so its saved inference artifact does not imply
            # that it contains a resumable optimizer state.
            compile_model=not args.cache_features,
            learning_rate=args.initial_learning_rate,
        )
        if model.output_shape[-1] != 2 or model.get_layer(config.BACKBONE_NAME).trainable:
            raise RuntimeError("expected two-class softmax with frozen backbone")

        callbacks = make_callbacks(
            run_dir,
            min_learning_rate=args.min_learning_rate,
            reduce_lr_factor=args.reduce_lr_factor,
            reduce_lr_patience=args.reduce_lr_patience,
        )
        metadata["status"] = "training"
        _json_write(run_dir / "metadata.json", metadata)
        if args.cache_features:
            print("Extracting frozen training features...", flush=True)
            train_data = _label_dataset(
                build_tf_dataset(train_frame, batch_size=args.batch_size, training=False, augment=False)
            )
            val_data = _label_dataset(
                build_tf_dataset(val_frame, batch_size=args.batch_size, training=False, augment=False)
            )
            x_train, y_train = extract_pooled_features(model, train_data)
            print("Extracting frozen validation features...", flush=True)
            x_val, y_val = extract_pooled_features(model, val_data)
            fit_model = build_cached_head(model, learning_rate=args.initial_learning_rate)
            history = fit_model.fit(
                x_train,
                y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(x_val, y_val),
                class_weight=weights,
                shuffle=True,
                callbacks=callbacks,
                verbose=2,
            )
        else:
            train_data = _label_dataset(
                build_tf_dataset(
                    train_frame,
                    batch_size=args.batch_size,
                    training=True,
                    augment=args.augment,
                    seed=config.RANDOM_SEED,
                )
            )
            val_data = _label_dataset(
                build_tf_dataset(val_frame, batch_size=args.batch_size, training=False, augment=False)
            )
            fit_model = model
            history = fit_model.fit(
                train_data,
                epochs=args.epochs,
                validation_data=val_data,
                class_weight=weights,
                callbacks=callbacks,
                verbose=2,
            )
        if len(history.epoch) != args.epochs:
            raise RuntimeError("training stopped before the requested number of epochs")
        # Keras CSVLogger can emit CRCRLF on Windows because csv.writer and
        # text-mode newline conversion both add a carriage return.
        history_path = run_dir / "history.csv"
        history_path.write_bytes(
            history_path.read_bytes().replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
        )
        model.save(run_dir / "final_model.keras")
        fit_model.load_weights(run_dir / "best_trainable.weights.h5")
        # Cached head and image model share the exact same classifier weights.
        model.save(run_dir / "best_model.keras")
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        metadata["result"] = {
            "completed_epochs": len(history.epoch),
            "best_epoch": best_epoch,
            "best_val_loss": float(history.history["val_loss"][best_epoch - 1]),
            "best_val_sparse_categorical_accuracy": float(
                history.history["val_sparse_categorical_accuracy"][best_epoch - 1]
            ),
            "last_epoch_learning_rate": float(history.history["learning_rate"][-1]),
        }
        metadata["status"] = "completed"
        _json_write(run_dir / "metadata.json", metadata)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        _json_write(run_dir / "metadata.json", metadata)
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.DENSENET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DENSENET_BATCH_SIZE)
    parser.add_argument(
        "--initial-learning-rate", type=float, default=config.DENSENET_INITIAL_LEARNING_RATE
    )
    parser.add_argument(
        "--min-learning-rate", type=float, default=config.DENSENET_MINIMUM_LEARNING_RATE
    )
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--weights", choices=("imagenet", "none"), default=config.BACKBONE_WEIGHTS)
    parser.add_argument("--train-limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--val-limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/section3a"))
    args = parser.parse_args()
    run_dir = train(args)
    print(f"Training complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
