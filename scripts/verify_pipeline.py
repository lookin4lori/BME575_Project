import argparse
import json
from datetime import datetime

import matplotlib

# Save figures to files without needing a pop-up plotting window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from scripts.classifier import build_classifier, compile_classifier
from scripts.dataset import build_tf_dataset, load_splits
from scripts.model import INPUT_SHAPE, count_parameters


def require(condition, message): ##jsut to check for failure
    if not condition:
        raise AssertionError(message)


def check_forward_pass(model, images, labels): #check probabilities
    require(
        tuple(images.shape[1:]) == INPUT_SHAPE,
        f"Unexpected image shape: {images.shape}",
    )

    require(np.isfinite(images.numpy()).all(), #check
        "Images contain NaN or infinite values.",
    )

    require(
        tuple(labels.shape) == (len(images),),
        f"Expected one scalar class label per image; got {labels.shape}",
    )

    require( np.isin(labels.numpy(), [0, 1]).all(),
        "Labels must contain only 0 and 1.",
    )

    # training=False disables dropout during this prediction check.
    predictions = model(images, training=False).numpy()

    require( predictions.shape == (len(images), 2),
        f"Expected two class probabilities per image; got {predictions.shape}",
    )

    require( np.isfinite(predictions).all(),
        "Predictions contain NaN or infinite values.",
    )

    require(
        ((predictions >= 0) & (predictions <= 1)).all(),
        "Softmax probabilities must be between 0 and 1.",
    )

    require(
        np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5),
        "Softmax probabilities must sum to 1 for each image.",
    )

    return {
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "prediction_shape": list(predictions.shape),
        "prediction_columns": ["no_dr", "any_dr"],
    }


def check_augmentation(splits): ##check traina augmentation and eval
    changed_views = 0
    attempted_views = 0

    # Use one image at a time so shuffling cannot confuse the comparison.
    for index in range(min(4, len(splits["train"]))):
        frame = splits["train"].iloc[[index]]

        original_images, original_labels = next(
            iter(build_tf_dataset(frame, training=False))
        )

        # seeded augmentation
        # Different seeds let us examine several possible transformations.
        for seed in range(config.RANDOM_SEED, config.RANDOM_SEED + 8):
            augmented_images, augmented_labels = next(
                iter(
                    build_tf_dataset(
                        frame,
                        training=True,
                        augment=True,
                        seed=seed,
                    )
                )
            )

            require(
                np.array_equal(
                    original_labels.numpy(),
                    augmented_labels.numpy(),
                ),
                "Augmentation changed an image label.",
            )

            changed_views += int(
                not np.array_equal(
                    original_images.numpy(),
                    augmented_images.numpy(),
                )
            )
            attempted_views += 1

    require(
        changed_views > 0,
        "No training-image changes were observed across the tested seeds.",
    )

    for split_name in ("val", "test"):
        frame = splits[split_name].head(4)

        dataset = build_tf_dataset(frame, training=False)

        first_images, first_labels = next(iter(dataset))
        second_images, second_labels = next(iter(dataset))

        require(
            np.array_equal(first_images.numpy(), second_images.numpy()),
            f"{split_name} images changed between repeated loads.",
        )

        require(
            np.array_equal(first_labels.numpy(), second_labels.numpy()),
            f"{split_name} labels changed between repeated loads.",
        )

        # The loader reject augmentation in evaluation .
        try:
            build_tf_dataset(frame, training=False, augment=True)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Augmentation was allowed in {split_name} evaluation mode."
            )

    return {
        "changed_training_views": changed_views,
        "attempted_training_views": attempted_views,
        "validation_and_test_repeatable": True,
        "evaluation_augmentation_rejected": True,
    }


def select_twenty_images(train_frame):
    """Select 10 no-DR and 10 any-DR images from TRAINING ONLY."""

    selected_groups = []

    for label in (0, 1):
        group = train_frame[
            train_frame["binary_label"] == label
        ].sort_values(config.ID_COLUMN)

        require(
            len(group) >= 10,
            f"Not enough training images for binary class {label}.",
        )

        selected_groups.append(
            group.sample(n=10, random_state=config.RANDOM_SEED)
        )

    return (
        pd.concat(selected_groups)
        .sample(frac=1, random_state=config.RANDOM_SEED)
        .reset_index(drop=True)
    )


def check_weight_updates(model, images, labels):
    """Confirm the head changes after training but the backbone does not."""

    backbone = model.get_layer(config.BACKBONE_NAME)
    head = model.layers[-1]

    counts = {
        "full_model": count_parameters(model),
        "backbone": count_parameters(backbone),
    }

    require(
        not backbone.trainable_weights,
        "The default backbone should be frozen.",
    )

    require(
        {id(w) for w in model.trainable_weights}
        == {id(w) for w in head.trainable_weights},
        "Trainable weights were found outside the classifier head.",
    )

    # Global pooling yields 1,024 features. Two softmax units have
    # 1,024 x 2 weights and two biases.
    require(
        counts["full_model"]["trainable"] == 2050,
        "Expected 2,050 trainable parameters for this classifier head.",
    )

    initial_model_weights = model.get_weights()
    initial_backbone_weights = backbone.get_weights()
    initial_head_weights = head.get_weights()

    ##one optimizer update using the real training images.
    model.train_on_batch(images, labels)

    require(
        all(
            np.array_equal(before, after)
            for before, after in zip(
                initial_backbone_weights,
                backbone.get_weights(),
            )
        ),
        "Frozen backbone weights or batch-normalization state changed.",
    )

    require(
        any(
            not np.array_equal(before, after)
            for before, after in zip(
                initial_head_weights,
                head.get_weights(),
            )
        ),
        "The classifier head did not update.",
    )

    # learning experiment starts fresh by undoing prpbe
    model.set_weights(initial_model_weights)

    # Reset optimizer state and model weights.
    compile_classifier(model)

    return counts


def run_overfit_test(model, images, labels, epochs, output_dir):
    """Repeatedly train on the same 20 images and save the learning curve."""

    history = []

    for epoch in range(epochs + 1):
        # Epoch 0 records the model BEFORE any diagnostic training.
        if epoch > 0:
            model.reset_metrics()
            model.train_on_batch(images, labels)

        # Evaluate the SAME training subset with dropout disabled.
        predictions = model(images, training=False)

        loss = float(tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                tf.cast(labels, tf.int32), predictions
            )
        ))

        accuracy = float(
            np.mean(
                np.argmax(predictions.numpy(), axis=1) == labels.numpy()
            )
        )

        require(np.isfinite(loss), "The diagnostic loss became non-finite.")

        history.append(
            {
                "epoch": epoch,
                "subset_loss": loss,
                "subset_accuracy": accuracy,
            }
        )

        # Save after each epoch
        pd.DataFrame(history).to_csv(
            output_dir / "overfit20_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={loss:.4f}, accuracy={accuracy:.3f}",
            flush=True,
        )

    history_frame = pd.DataFrame(history)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        history_frame["epoch"],
        history_frame["subset_loss"],
    )
    axes[0].set(xlabel="Epoch", ylabel="Training-subset loss")

    axes[1].plot(
        history_frame["epoch"],
        history_frame["subset_accuracy"],
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Training-subset accuracy",
        ylim=(0, 1.05),
    )

    figure.suptitle(
        "20 training images: no augmentation; inference-mode metrics"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "overfit20_curve.png", dpi=150)
    plt.close(figure)

    return {
        "initial": history[0],
        "final": history[-1],
        "loss_decreased": (
            history[-1]["subset_loss"] < history[0]["subset_loss"]
        ),
        "reached_95_percent": history[-1]["subset_accuracy"] >= 0.95,
        "note": (
            "95% is a diagnostic target, not a paper requirement. "
            "These are training-subset results, not held-out performance."
        ),
    }


def run_verification(epochs, output_dir, report):
    """Connect and run all five verification tasks."""

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)


    splits = load_splits(check_images=True)

    # Use the actual pretrained model wit da backbone frozen.
    model = build_classifier(unfreeze_last_n_blocks=0)

    report["settings"] = {
        "seed": config.RANDOM_SEED,
        "pretrained_weights": config.BACKBONE_WEIGHTS,
        "learning_rate": config.DENSENET_INITIAL_LEARNING_RATE,
        "dropout": config.CLASSIFIER_DROPOUT_RATE,
        "output_units": config.CLASSIFIER_OUTPUT_UNITS,
        "output_activation": config.CLASSIFIER_OUTPUT_ACTIVATION,
        "loss": config.CLASSIFIER_LOSS,
        "prediction_columns": ["no_dr", "any_dr"],
        "diagnostic_epochs": epochs,
        "configured_batch_size": config.DENSENET_BATCH_SIZE,
        "effective_diagnostic_batch_size": 20,
        "overfit_augmentation": False,
        "overfit_class_weights": None,
        "tensorflow_version": tf.__version__,
    }

    # 1. Artificial images test the model interface without file loading
    dummy_images = tf.zeros((2, *INPUT_SHAPE))
    dummy_labels = tf.zeros(2)

    report["dummy_batch"] = check_forward_pass(
        model, dummy_images, dummy_labels
    )
    print("PASS: dummy-batch forward pass", flush=True)

    # 2. Real images test dataset.py -> classifier.py.
    # Validation/test images are used for shape checks NOT training.
    report["real_batches"] = {}

    for split_name, frame in splits.items():
        dataset = build_tf_dataset(
            frame.head(4),
            training=(split_name == "train"),
        )
        images, labels = next(iter(dataset))

        report["real_batches"][split_name] = check_forward_pass(
            model, images, labels
        )

    print("PASS: real-image end-to-end forward passes", flush=True)

    # 3. Confirm the augmentation policy.
    report["augmentation"] = check_augmentation(splits)
    print("PASS: augmentation checks", flush=True)

    # Prepare the fixed 20-image diagnostic subset
    subset = select_twenty_images(splits["train"])

    # Save portable IDs/labels, without  image paths
    subset.drop(columns=["image_path"], errors="ignore").to_csv(
        output_dir / "overfit20_ids.csv",
        index=False,
    )

    # training=False here means no augmentation and no shuffle.
    # train_on_batch below still trains the MODEL with dropout active.
    diagnostic_dataset = build_tf_dataset(
        subset,
        batch_size=20,
        training=False,
    )
    images, labels = next(iter(diagnostic_dataset))

    # Sparse categorical cross-entropy consumes scalar 0/1 class labels.
    labels = tf.cast(tf.reshape(labels, (-1,)), tf.int32)

    # 4. Verify both declared trainability and actual weight updates.
    report["parameters"] = check_weight_updates(model, images, labels)
    report["weight_updates"] = "Head changed; frozen backbone unchanged."
    print("PASS: parameter and weight-update checks", flush=True)

    # 5. Test learning on the small training subset.
    report["overfit20"] = run_overfit_test(
        model, images, labels, epochs, output_dir
    )

    report["status"] = "Checks passed; learning diagnostic completed."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DENSENET_EPOCHS,
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # A unique folder prevents a new run from overwriting an older run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = config.REPORT_DIR / "verification" / timestamp
    output_dir.mkdir(parents=True)

    report = {
        "purpose": "Section 2c engineering verification",
        "status": "Started",
    }

    try:
        run_verification(args.epochs, output_dir, report)
    except Exception as error:
        report["status"] = "Failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Save the report even if a check raises an error.
        with (output_dir / "verification.json").open("w") as file:
            json.dump(report, file, indent=2)

        print(f"\nResults saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
