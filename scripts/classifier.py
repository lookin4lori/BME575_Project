"""Binary diabetic-retinopathy classifier head (Section 2b).

The Section 1c input pipeline already applies DenseNet ``preprocess_input``
and yields scalar float32 labels (0 = no DR, 1 = any DR). This module adds a
global average pooling layer, optional dropout, and one sigmoid output to the
Section 2a backbone. It must not normalize inputs a second time.

The paper supplies DenseNet121, global average pooling, binary classification,
Adam, and the training settings in ``config.py``. The exact loss and dropout
rate are team implementation choices; the paper does not prescribe them.
"""

from __future__ import annotations

import math

import tensorflow as tf

import config
from scripts.model import INPUT_SHAPE, build_backbone, set_backbone_trainable


def _positive_learning_rate(learning_rate: float) -> float:
    """Convert a requested learning rate to a finite, positive float."""
    try:
        value = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("learning_rate must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("learning_rate must be a finite positive number")
    return value


def compile_classifier(
    classifier: tf.keras.Model,
    *,
    learning_rate: float = config.DENSENET_INITIAL_LEARNING_RATE,
) -> tf.keras.Model:
    """Compile with a new Adam optimizer and binary cross-entropy loss.

    Call again after changing which backbone layers are trainable. A new
    optimizer avoids reusing state tied to the former set of trainable weights.
    """
    rate = _positive_learning_rate(learning_rate)
    optimizer = tf.keras.optimizers.get(config.CLASSIFIER_OPTIMIZER)
    optimizer.learning_rate = rate
    classifier.compile(
        optimizer=optimizer,
        loss=config.CLASSIFIER_LOSS,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return classifier


def build_classifier(
    *,
    weights: str | None = config.BACKBONE_WEIGHTS,
    unfreeze_last_n_blocks: int = config.UNFREEZE_LAST_N_BLOCKS,
    dropout_rate: float = config.CLASSIFIER_DROPOUT_RATE,
    compile_model: bool = True,
    learning_rate: float = config.DENSENET_INITIAL_LEARNING_RATE,
) -> tf.keras.Model:
    """Build the Section 2a DenseNet121 backbone with a binary head.

    ``weights=None`` is useful for offline tests. By default the backbone is
    frozen while the classifier's final dense layer can learn. Dropout may be
    disabled with ``dropout_rate=0.0``. If constructing an unfrozen backbone,
    pass an explicit learning rate suited to fine-tuning; the default 0.01 is
    the paper's initial rate for the frozen-feature setup.
    """
    try:
        rate = float(dropout_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("dropout_rate must be in [0, 1)") from exc
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("dropout_rate must be in [0, 1)")
    if compile_model:
        _positive_learning_rate(learning_rate)

    backbone = build_backbone(
        weights=weights,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    image = tf.keras.Input(shape=INPUT_SHAPE, name="fundus_image")
    # Inference mode keeps pretrained BatchNorm statistics fixed, including
    # when the final dense block is unfrozen for a later fine-tuning stage.
    features = backbone(image, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    if rate:
        features = tf.keras.layers.Dropout(rate, name="classifier_dropout")(features)
    probability = tf.keras.layers.Dense(
        config.CLASSIFIER_OUTPUT_UNITS,
        activation=config.CLASSIFIER_OUTPUT_ACTIVATION,
        name="dr_probability",
    )(features)
    classifier = tf.keras.Model(image, probability, name="dr_binary_classifier")
    if compile_model:
        compile_classifier(classifier, learning_rate=learning_rate)
    return classifier


def set_classifier_backbone_trainable(
    classifier: tf.keras.Model,
    unfreeze_last_n_blocks: int,
    *,
    learning_rate: float | None = None,
) -> tf.keras.Model:
    """Freeze or unfreeze the last N dense blocks and recompile the classifier.

    ``0`` freezes the entire backbone. ``1`` through ``4`` unfreeze that many
    final dense blocks; BatchNorm remains frozen as configured in Section 2a.
    An explicit learning rate is recommended when starting fine-tuning, e.g.
    ``config.DENSENET_MINIMUM_LEARNING_RATE``. If omitted, keep the compiled
    model's current rate, or use the initial rate if it was not compiled.
    """
    backbone = classifier.get_layer(config.BACKBONE_NAME)
    if learning_rate is None:
        optimizer = getattr(classifier, "optimizer", None)
        if optimizer is None:
            rate = config.DENSENET_INITIAL_LEARNING_RATE
        else:
            rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
    else:
        rate = learning_rate
    rate = _positive_learning_rate(rate)
    set_backbone_trainable(backbone, unfreeze_last_n_blocks)
    return compile_classifier(classifier, learning_rate=rate)
