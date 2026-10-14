"""Tests for the binary DR classifier built on the Section 2a backbone."""

import math

import pytest

tf = pytest.importorskip("tensorflow")

from scripts import classifier as c


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


def test_head_is_global_average_pooling_optional_dropout_and_one_sigmoid(binary_classifier):
    head = binary_classifier.layers[-3:]
    assert isinstance(head[0], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(head[1], tf.keras.layers.Dropout)
    assert head[1].rate == pytest.approx(0.5)
    assert isinstance(head[2], tf.keras.layers.Dense)
    assert head[2].units == 1
    assert head[2].activation is tf.keras.activations.sigmoid
    assert binary_classifier.output_shape == (None, 1)

    probabilities = binary_classifier(tf.zeros((2, 224, 224, 3)), training=False)
    assert probabilities.shape == (2, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(probabilities)))
    assert bool(tf.reduce_all((probabilities >= 0) & (probabilities <= 1)))


def test_compile_uses_adam_binary_crossentropy_and_requested_learning_rate(binary_classifier):
    assert isinstance(binary_classifier.optimizer, tf.keras.optimizers.Adam)
    assert binary_classifier.loss == "binary_crossentropy" or isinstance(
        binary_classifier.loss, tf.keras.losses.BinaryCrossentropy
    )
    assert _learning_rate(binary_classifier) == pytest.approx(0.002)

    previous_optimizer = binary_classifier.optimizer
    c.compile_classifier(binary_classifier, learning_rate=0.001)
    assert isinstance(binary_classifier.optimizer, tf.keras.optimizers.Adam)
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


def test_dropout_can_be_omitted():
    model = c.build_classifier(weights=None, dropout_rate=0.0, compile_model=False)
    assert not any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)
    assert isinstance(model.layers[-2], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(model.layers[-1], tf.keras.layers.Dense)
    assert model.output_shape == (None, 1)
    assert getattr(model, "optimizer", None) is None

    # Trainability can also be configured before the first explicit compile.
    c.set_classifier_backbone_trainable(model, 0)
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


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
