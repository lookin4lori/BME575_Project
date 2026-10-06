"""Pretrained backbone for the APTOS 2019 DR project (Section 2a).

Builds DenseNet121 with ImageNet weights and its original classifier removed.
It is frozen by default to match the paper, which used pretrained feature
detectors and trained only the classifier. Section 2b adds pooling, dropout,
and the output layer on top.

Inputs must already be normalized with
tf.keras.applications.densenet.preprocess_input. dataset.py does this, so the
model must not normalize again.

Run from the repo root with: python -m scripts.model
"""

import argparse
import json

import numpy as np
import tensorflow as tf

import config

# Supported backbones. Comparison models for Section 3c can be added here.
# Note: other models may need a different preprocess_input than dataset.py uses.
BACKBONES = {
    "densenet121": tf.keras.applications.DenseNet121,
}

# DenseNet121's four dense blocks, named conv2_ to conv5_ in Keras
# (6, 12, 24, and 16 dense layers).
DENSE_BLOCK_STAGES = (2, 3, 4, 5)

INPUT_SHAPE = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS)

#load dense net with weights
def build_backbone(name=config.BACKBONE_NAME, weights=config.BACKBONE_WEIGHTS,
                   unfreeze_last_n_blocks=config.UNFREEZE_LAST_N_BLOCKS):
    """Build the pretrained backbone without its top layer.

    For DenseNet121 with 224x224 input, the output is a 7x7x1024 feature map.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Options: {sorted(BACKBONES)}")
    backbone = BACKBONES[name](include_top=False, weights=weights, input_shape=INPUT_SHAPE)
    set_backbone_trainable(backbone, unfreeze_last_n_blocks)
    return backbone

#controls layers that can learn
def set_backbone_trainable(backbone, unfreeze_last_n_blocks=0,
                           keep_batchnorm_frozen=config.KEEP_BATCHNORM_FROZEN):
    """Freeze the backbone, or unfreeze only its last N dense blocks for fine-tuning.

    0 freezes everything (the paper's setup). 1 unfreezes conv5 only, 2 unfreezes
    conv4 and conv5, and so on up to 4. BatchNorm layers stay frozen by default,
    which is standard practice when fine-tuning a pretrained model.
    """
    n = unfreeze_last_n_blocks
    if not 0 <= n <= len(DENSE_BLOCK_STAGES):
        raise ValueError(f"unfreeze_last_n_blocks must be 0-{len(DENSE_BLOCK_STAGES)}, got {n}")

    if n == 0:
        backbone.trainable = False
        return backbone

    # The outer switch must be on for the per-layer settings below to take effect.
    backbone.trainable = True

    first_stage = DENSE_BLOCK_STAGES[-n]
    start_prefix = f"conv{first_stage}_block1_"
    start = next(i for i, layer in enumerate(backbone.layers)
                 if layer.name.startswith(start_prefix))

    for i, layer in enumerate(backbone.layers):
        trainable = i >= start
        if keep_batchnorm_frozen and isinstance(layer, tf.keras.layers.BatchNormalization):
            trainable = False
        layer.trainable = trainable
    return backbone

#adds up weights op will update and wont
def count_parameters(model):
    """Return the total, trainable, and non-trainable parameter counts."""
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"total": trainable + frozen, "trainable": trainable, "non_trainable": frozen}

#sends seros through backbone to check shape
def check_output_shape(backbone, batch_size=2):
    """Run a dummy batch through the backbone and confirm the output shape."""
    dummy = tf.zeros((batch_size, *INPUT_SHAPE))
    output = backbone(dummy, training=False)
    expected = (batch_size,) + tuple(backbone.output_shape[1:])
    actual = tuple(output.shape)
    if actual != expected:
        raise AssertionError(f"Expected output shape {expected}, got {actual}")
    return actual

#builds model, runs shape check, freeze level and saves etc etc
def main():
    """Build the backbone, check shapes, and save parameter counts for the report."""
    parser = argparse.ArgumentParser(description="Build and inspect the model backbone.")
    parser.add_argument("--full-summary", action="store_true",
                        help="Also save the full layer-by-layer Keras summary.")
    args = parser.parse_args()

    backbone = build_backbone()
    print(f"Backbone: {config.BACKBONE_NAME} (weights: {config.BACKBONE_WEIGHTS})")
    print(f"Input shape:  {INPUT_SHAPE}")
    print(f"Output shape: {tuple(backbone.output_shape[1:])}")
    print(f"Layers: {len(backbone.layers)}")

    out_shape = check_output_shape(backbone)
    print(f"Dummy-batch check passed: {out_shape}\n")

    rows = []
    for n in range(len(DENSE_BLOCK_STAGES) + 1):
        set_backbone_trainable(backbone, n)
        rows.append({"unfreeze_last_n_blocks": n, **count_parameters(backbone)})
    set_backbone_trainable(backbone, config.UNFREEZE_LAST_N_BLOCKS)  # restore the default

    print(f"{'unfrozen blocks':>16} {'total':>12} {'trainable':>12} {'non-trainable':>14}")
    for r in rows:
        print(f"{r['unfreeze_last_n_blocks']:>16} {r['total']:>12,} "
              f"{r['trainable']:>12,} {r['non_trainable']:>14,}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.REPORT_DIR / "model_backbone_summary.json"
    summary_path.write_text(json.dumps({
        "backbone": config.BACKBONE_NAME,
        "weights": config.BACKBONE_WEIGHTS,
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(backbone.output_shape[1:]),
        "num_layers": len(backbone.layers),
        "default_unfreeze_last_n_blocks": config.UNFREEZE_LAST_N_BLOCKS,
        "keep_batchnorm_frozen": config.KEEP_BATCHNORM_FROZEN,
        "parameter_counts": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {summary_path}")

    if args.full_summary:
        full_path = config.REPORT_DIR / "model_backbone_full_summary.txt"
        with open(full_path, "w", encoding="utf-8") as f:
            backbone.summary(print_fn=lambda line, *a, **k: f.write(line + "\n"))
        print(f"Saved: {full_path}")


if __name__ == "__main__":
    main()