"""Tests for the paper-aligned, two-class DenseNet121 classifier head."""

import math

import pytest

tf = pytest.importorskip("tensorflow")

from scripts import classifier as c


def _assert_adam(optimizer):
    """Accept modern and legacy Adam classes across supported Keras versions."""
    adam_types = [tf.keras.optimizers.Adam]
    legacy = getattr(tf.keras.optimizers, "legacy", None)
    legacy_adam = getattr(legacy, "Adam", None)
    if isinstance(legacy_adam, type):
        adam_types.append(legacy_adam)
    assert isinstance(optimizer, tuple(adam_types))


def _backbone(model):
    """Find the DenseNet nested inside the full classifier model."""
    nested = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    assert len(nested) == 1
    return nested[0]


def _learning_rate(model):
    return float(tf.keras.backend.get_value(model.optimizer.learning_rate))


@pytest.fixture(scope="module")
def binary_classifier():
    # Do not download pretrained weights in tests; the architecture is identical.
    return c.build_classifier(
        weights=None,
        unfreeze_last_n_blocks=0,
        dropout_rate=0.5,
        learning_rate=0.002,
    )


def test_head_is_global_average_pooling_optional_dropout_and_two_softmax(binary_classifier):
    head = binary_classifier.layers[-3:]
    assert isinstance(head[0], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(head[1], tf.keras.layers.Dropout)
    assert head[1].rate == pytest.approx(0.5)
    assert isinstance(head[2], tf.keras.layers.Dense)
    assert head[2].units == 2
    assert head[2].activation is tf.keras.activations.softmax
    assert binary_classifier.output_shape == (None, 2)

    probabilities = binary_classifier(tf.zeros((2, 224, 224, 3)), training=False)
    assert probabilities.shape == (2, 2)
    assert bool(tf.reduce_all(tf.math.is_finite(probabilities)))
    assert bool(tf.reduce_all((probabilities >= 0) & (probabilities <= 1)))
    tf.debugging.assert_near(
        tf.reduce_sum(probabilities, axis=-1),
        tf.ones((2,)),
        atol=1e-6,
    )


def test_compile_uses_adam_sparse_crossentropy_and_requested_learning_rate(binary_classifier):
    _assert_adam(binary_classifier.optimizer)
    assert binary_classifier.loss == "sparse_categorical_crossentropy" or isinstance(
        binary_classifier.loss, tf.keras.losses.SparseCategoricalCrossentropy
    )
    assert _learning_rate(binary_classifier) == pytest.approx(0.002)

    # Integer labels 0 = no DR and 1 = any DR must work with the two outputs.
    evaluation = binary_classifier.evaluate(
        tf.zeros((2, 224, 224, 3)),
        tf.constant([0, 1], dtype=tf.int32),
        verbose=0,
        return_dict=True,
    )
    assert math.isfinite(evaluation["loss"])
    assert "sparse_categorical_accuracy" in evaluation

    previous_optimizer = binary_classifier.optimizer
    c.compile_classifier(binary_classifier, learning_rate=0.001)
    _assert_adam(binary_classifier.optimizer)
    assert binary_classifier.optimizer is not previous_optimizer
    assert _learning_rate(binary_classifier) == pytest.approx(0.001)


def test_frozen_backbone_keeps_classifier_head_trainable(binary_classifier):
    c.set_classifier_backbone_trainable(binary_classifier, 0, learning_rate=0.002)
    backbone = _backbone(binary_classifier)
    assert not backbone.trainable
    assert not backbone.trainable_weights
    assert binary_classifier.layers[-1].trainable_weights
    assert {id(weight) for weight in binary_classifier.trainable_weights} == {
        id(weight) for weight in binary_classifier.layers[-1].trainable_weights
    }


def test_unfreeze_and_refreeze_recompile_with_fresh_optimizer(binary_classifier):
    c.set_classifier_backbone_trainable(binary_classifier, 0, learning_rate=0.002)
    backbone = _backbone(binary_classifier)
    frozen_optimizer = binary_classifier.optimizer

    c.set_classifier_backbone_trainable(binary_classifier, 1, learning_rate=0.00005)
    assert binary_classifier.optimizer is not frozen_optimizer
    assert _learning_rate(binary_classifier) == pytest.approx(0.00005)
    trainable_weighted_layers = [
        layer for layer in backbone.layers if layer.trainable and layer.trainable_weights
    ]
    assert trainable_weighted_layers
    assert all(layer.name.startswith("conv5_") for layer in trainable_weighted_layers)
    batchnorm_layers = [
        layer for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    assert batchnorm_layers and not any(layer.trainable for layer in batchnorm_layers)

    unfrozen_optimizer = binary_classifier.optimizer
    c.set_classifier_backbone_trainable(binary_classifier, 0, learning_rate=0.002)
    assert binary_classifier.optimizer is not unfrozen_optimizer
    assert _learning_rate(binary_classifier) == pytest.approx(0.002)
    assert not backbone.trainable_weights
    assert binary_classifier.layers[-1].trainable_weights


def test_default_head_omits_paper_unspecified_dropout():
    model = c.build_classifier(weights=None, compile_model=False)
    assert not any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)
    assert isinstance(model.layers[-2], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(model.layers[-1], tf.keras.layers.Dense)
    assert model.layers[-1].units == 2
    assert model.layers[-1].activation is tf.keras.activations.softmax
    assert model.output_shape == (None, 2)
    assert getattr(model, "optimizer", None) is None

    # Trainability can also be configured before the first explicit compile.
    c.set_classifier_backbone_trainable(model, 0)
    _assert_adam(model.optimizer)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.0, math.nan])
def test_invalid_dropout_rate_is_rejected(bad_rate):
    with pytest.raises(ValueError, match="dropout"):
        c.build_classifier(weights=None, dropout_rate=bad_rate, compile_model=False)


@pytest.mark.parametrize("bad_rate", [0.0, -0.001, math.nan])
def test_invalid_learning_rate_is_rejected(binary_classifier, bad_rate):
    with pytest.raises(ValueError, match="learning_rate"):
        c.compile_classifier(binary_classifier, learning_rate=bad_rate)


@pytest.mark.parametrize("bad_blocks", [-1, 5])
def test_invalid_unfreeze_block_count_is_rejected(binary_classifier, bad_blocks):
    with pytest.raises(ValueError, match="unfreeze_last_n_blocks"):
        c.set_classifier_backbone_trainable(binary_classifier, bad_blocks)
