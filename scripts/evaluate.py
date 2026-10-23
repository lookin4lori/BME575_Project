import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

import config
from scripts.dataset import load_splits, build_tf_dataset


def sha256_file(path): ##records file so we can record exactly while model was used
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(path, expected_hash): #checks that split files same as kevins even tho windows and mac have diff line endings (i have a mac)
    original = Path(path).read_bytes()
    lf = original.replace(b"\r\n", b"\n")

    versions = {
        "exact": original,
        "LF": lf,
        "CRLF": lf.replace(b"\n", b"\r\n"),
    }

    for format_name, contents in versions.items():
        if hashlib.sha256(contents).hexdigest() == expected_hash:
            return format_name

    raise ValueError(
        f"{path} differs from the training manifest, beyond line endings."
    )


def predict_frame(model, frame, batch_size): #loads images in order, gets predictions formaat and attaches them to IDs
    dataset = build_tf_dataset(frame, batch_size=batch_size, training=False, augment=False,)

    probability_batches = []
    label_batches = []

    for images, labels in dataset:
        # dataset.py already normalizes the images.
        # Inference mode prevents training-time behavior.
        output = model(images, training=False)

        probability_batches.append(np.asarray(output))
        label_batches.append(np.asarray(labels).reshape(-1))

    probabilities = np.concatenate(probability_batches)
    observed_labels = np.concatenate(label_batches)
    expected_labels = frame["binary_label"].to_numpy(dtype=np.int32)

    if not np.array_equal(observed_labels, expected_labels):
        raise ValueError("Dataset labels are not in manifest order.")

    if probabilities.shape != (len(frame), 2):
        raise ValueError("Expected two probabilities per image.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinity.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between zero and one.")

    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Softmax probabilities must sum to one.")

    results = frame[["id_code", "diagnosis", "binary_label"]].copy()
    results = results.rename(columns={"binary_label": "true_label"})
    results["probability_no_dr"] = probabilities[:, 0]
    results["probability_dr"] = probabilities[:, 1]
    results["predicted_label"] = probabilities.argmax(axis=1)
    results["correct"] = (
        results["predicted_label"] == results["true_label"]
    )

    return results


def calculate_metrics(predictions): #this is to compare predictions wit true labels and cal performa
    labels = predictions["true_label"].to_numpy(dtype=np.int32)
    predicted = predictions["predicted_label"].to_numpy()
    probabilities = predictions[
        ["probability_no_dr", "probability_dr"]
    ].to_numpy()

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Full-split ROC-AUC requires both binary classes.")

    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    loss = float(
        tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, probabilities
            )
        ).numpy()
    )

    return {
        "number_of_images": int(len(labels)),
        "loss_unweighted": loss,
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity_dr": float(tp / (tp + fn)),
        "specificity_no_dr": float(tn / (tn + fp)),
        "roc_auc_dr": float(
            roc_auc_score(labels, probabilities[:, 1])
        ),
        "confusion_matrix": matrix.tolist(),
        "matrix_order": ["No DR", "Any DR"],
        "matrix_rows": "true labels",
        "matrix_columns": "predicted labels",
        "classification_report": classification_report(
            labels,
            predicted,
            labels=[0, 1],
            target_names=["No DR", "Any DR"],
            output_dict=True,
            zero_division=0,
        ),
    }


def make_subgroups(predictions): #serpeates bw OG severitiy and counts correct/incorrect
    rows = []

    for grade in range(5):
        group = predictions[predictions["diagnosis"] == grade]
        count = len(group)
        correct = int(group["correct"].sum())

        rows.append({
            "original_grade": grade,
            "true_binary_label": int(grade > 0),
            "number_of_images": count,
            "correct": correct,
            "incorrect": count - correct,
            "predicted_no_dr": int(
                (group["predicted_label"] == 0).sum()
            ),
            "predicted_dr": int(
                (group["predicted_label"] == 1).sum()
            ),
            "fraction_correct": correct / count if count else None,
        })

    return pd.DataFrame(rows)


def write_json(path, contents): # saves a dictionty as jason file
    with open(path, "w") as file:
        json.dump(contents, file, indent=2, allow_nan=False)


def save_outputs(predictions, metrics, metadata, output_dir): #saves prediction tables,figures and metrics we will need
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    make_subgroups(predictions).to_csv(
        output_dir / "subgroup_counts.csv", index=False
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "evaluation_metadata.json", metadata)

    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(
        matrix,
        index=["True No DR", "True Any DR"],
        columns=["Predicted No DR", "Predicted Any DR"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        matrix, display_labels=["No DR", "Any DR"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{metadata['split']} split: binary DR classification")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, thresholds = roc_curve(
        predictions["true_label"],
        predictions["probability_dr"],
    )

    pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": thresholds,
    }).to_csv(output_dir / "roc_curve.csv", index=False)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc_dr']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(f"{metadata['split']} split: DR ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Record installed versions for reproducibility.
    packages = subprocess.check_output(
        [__import__("sys").executable, "-m", "pip", "freeze"],
        text=True,
    )
    (output_dir / "environment.txt").write_text(packages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODEL_DIR / "best_model.keras",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            config.REPORT_DIR
            / "training"
            / "20261019T213748_201239Z"
            / "metadata.json"
        ),
    )
    args = parser.parse_args()

    with open(args.metadata) as file:
        training = json.load(file)

    manifest_checks = {}
    for name in ["train", "val", "test"]:
        path = config.SPLIT_DIR / f"{name}.csv"
        match = check_manifest(path, training["manifest_sha256"][name])
        manifest_checks[name] = {
            "local_sha256": sha256_file(path),
            "training_sha256": training["manifest_sha256"][name],
            "matching_line_endings": match,
        }

    # Checks IDs, labels, and overlap without loading test image pixels.
    splits = load_splits(check_images=False)

    for name, frame in splits.items():
        if len(frame) != training["split_counts"][name]["size"]:
            raise ValueError(f"Unexpected number of images in {name}.")

    model = tf.keras.models.load_model(
        args.checkpoint, compile=False
    )

    if model.input_shape != (None, 224, 224, 3):
        raise ValueError(f"Unexpected input shape: {model.input_shape}")
    if model.output_shape != (None, 2):
        raise ValueError(f"Unexpected output shape: {model.output_shape}")
    if model.layers[-1].activation.__name__ != "softmax":
        raise ValueError("Expected a softmax output layer.")

    batch_size = int(training["settings"]["batch_size"])

    print("Checking validation performance...", flush=True)
    val_predictions = predict_frame(model, splits["val"], batch_size)
    val_metrics = calculate_metrics(val_predictions)

    expected = training["result"]
    expected_accuracy = expected["best_val_sparse_categorical_accuracy"]
    expected_loss = expected["best_val_loss"]

    print(f"Validation accuracy: {val_metrics['accuracy']:.6f}")
    print(f"Expected accuracy:   {expected_accuracy:.6f}")
    print(f"Validation loss:     {val_metrics['loss_unweighted']:.6f}")
    print(f"Expected loss:       {expected_loss:.6f}")

    # Project verification tolerances, not paper hyperparameters.
    # Do not widen them just to make an unexplained mismatch pass.
    if (
        abs(val_metrics["accuracy"] - expected_accuracy) > 1e-5
        or abs(val_metrics["loss_unweighted"] - expected_loss) > 1e-3
    ):
        raise ValueError(
            "Validation does not match the handover. "
            "Investigate before evaluating the test set."
        )

    if args.split == "val":
        predictions = val_predictions
        metrics = val_metrics
    else:
        print("Validation matched. Evaluating test split...", flush=True)
        predictions = predict_frame(model, splits["test"], batch_size)
        metrics = calculate_metrics(predictions)

    metadata = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "training_run_id": training["run_id"],
        "selected_epoch": expected["best_epoch"],
        "selection_criterion": "lowest validation loss",
        "training_metadata_sha256": sha256_file(args.metadata),
        "manifest_checks": manifest_checks,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "batch_size": batch_size,
        "augmentation": False,
        "shuffle": False,
        "normalization": "dataset.py DenseNet preprocess_input, once",
        "label_mapping": {"0": "No DR", "1": "Any DR"},
        "class_prediction": "argmax",
        "roc_probability_column": 1,
        "validation_accuracy": val_metrics["accuracy"],
        "validation_loss": val_metrics["loss_unweighted"],
        "validation_accuracy_tolerance": 1e-5,
        "validation_loss_tolerance": 1e-3,
        "scope": (
            "Evaluation of the team's paper-aligned baseline. "
            "Additional metrics and grade subgroups are assessment extensions. "
            "No fitting or test-based tuning."
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        config.REPORT_DIR / "evaluation" / f"{timestamp}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    save_outputs(predictions, metrics, metadata, output_dir)

    print(f"\nImages evaluated: {metrics['number_of_images']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"DR sensitivity: {metrics['sensitivity_dr']:.2%}")
    print(f"No-DR specificity: {metrics['specificity_no_dr']:.2%}")
    print(f"DR ROC-AUC: {metrics['roc_auc_dr']:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
