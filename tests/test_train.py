"""Small, data-free checks for the Section 3a training entry point.

The full 50-epoch APTOS run is intentionally outside unit tests. These tests
exercise the configuration guard, callbacks, and the frozen-feature path that
would otherwise be easy to get wrong without loading pretrained weights.
"""

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from scripts.classifier import build_classifier
from scripts import train


def _settings(**changes):
    settings = {
        "epochs": 50,
        "batch_size": 32,
        "initial_learning_rate": 0.01,
        "min_learning_rate": 0.00005,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,
        "augment": False,
        "cache_features": True,
    }
    settings.update(changes)
    return settings


@pytest.mark.parametrize(
    "bad_setting",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"initial_learning_rate": 0},
        {"min_learning_rate": 0},
        {"min_learning_rate": 0.02},
        {"reduce_lr_factor": 0},
        {"reduce_lr_factor": 1},
        {"reduce_lr_patience": 0},
        {"augment": True},  # Caching once cannot preserve per-epoch augmentation.
    ],
)
def test_invalid_training_settings_fail_before_training(bad_setting):
    with pytest.raises(ValueError):
        train.validate_training_settings(**_settings(**bad_setting))


def test_paper_training_settings_are_accepted():
    train.validate_training_settings(**_settings())
    train.validate_training_settings(**_settings(augment=True, cache_features=False))


def test_callbacks_record_history_best_checkpoint_and_minimum_lr(tmp_path):
    callbacks = train.make_callbacks(
        tmp_path,
        min_learning_rate=0.00005,
        reduce_lr_factor=0.5,
        reduce_lr_patience=3,
    )
    expected_types = (
        tf.keras.callbacks.CSVLogger,
        tf.keras.callbacks.ModelCheckpoint,
        tf.keras.callbacks.ReduceLROnPlateau,
        tf.keras.callbacks.TerminateOnNaN,
    )
    for callback_type in expected_types:
        assert sum(isinstance(callback, callback_type) for callback in callbacks) == 1

    csv_logger = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.CSVLogger))
    checkpoint = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
    lr_reducer = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
    assert Path(csv_logger.filename).parent == tmp_path
    assert Path(checkpoint.filepath).parent == tmp_path
    assert checkpoint.save_best_only
    assert checkpoint.monitor == "val_loss"
    assert lr_reducer.monitor == "val_loss"
    assert lr_reducer.factor == pytest.approx(0.5)
    assert lr_reducer.patience == 3
    assert lr_reducer.min_lr == pytest.approx(0.00005)


@pytest.fixture(scope="module")
def frozen_classifier_and_batch():
    # Same DenseNet121 architecture, with random weights to keep tests offline.
    model = build_classifier(weights=None, compile_model=False)
    images = tf.stack(
        [tf.zeros((224, 224, 3), dtype=tf.float32),
         tf.ones((224, 224, 3), dtype=tf.float32)]
    )
    labels = tf.constant([0.0, 1.0], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
    return model, dataset, images


def test_cached_features_match_frozen_backbone_and_keep_binary_labels(
    frozen_classifier_and_batch,
):
    model, dataset, images = frozen_classifier_and_batch
    features, labels = train.extract_pooled_features(model, dataset)

    assert features.shape == (2, 1024)
    assert features.dtype == np.float32
    assert labels.shape == (2,)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, [0, 1])
    assert np.isfinite(features).all()

    # The cache must contain the same pooled representation that the full
    # classifier sees, without retraining or normalizing the backbone again.
    backbone = model.get_layer("densenet121")
    manual = tf.reduce_mean(backbone(images, training=False), axis=(1, 2)).numpy()
    np.testing.assert_allclose(features, manual, rtol=1e-5, atol=1e-6)


def test_cached_head_shares_softmax_weights_with_full_classifier(
    frozen_classifier_and_batch,
):
    model, dataset, images = frozen_classifier_and_batch
    features, _ = train.extract_pooled_features(model, dataset)
    head = train.build_cached_head(model, learning_rate=0.01)

    assert head.input_shape == (None, 1024)
    assert head.output_shape == (None, 2)
    assert head.get_layer("dr_probability").kernel is model.get_layer("dr_probability").kernel
    assert isinstance(head.optimizer, tf.keras.optimizers.Adam)
    assert float(tf.keras.backend.get_value(head.optimizer.learning_rate)) == pytest.approx(0.01)
    assert head.loss == "sparse_categorical_crossentropy" or isinstance(
        head.loss, tf.keras.losses.SparseCategoricalCrossentropy
    )
    np.testing.assert_allclose(
        head(features, training=False).numpy(),
        model(images, training=False).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
