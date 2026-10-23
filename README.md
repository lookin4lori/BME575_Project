# BMME 575 DenseNet121 Project

Validation of a DenseNet121 diabetic-retinopathy classification
method using the APTOS 2019 dataset.

## Phase 1: data preparation

Phase 1 includes:

- Repository setup
- Train/validation/test splitting
- Leakage checks
- Class-distribution tables

## Section 2b: binary classifier head (Kaiyao)

`scripts/classifier.py` builds the classifier on the existing DenseNet121
backbone in `scripts/model.py`. The head follows the reference paper's §4.3:
global average pooling, a fully connected layer, and softmax. For this
two-class task it returns `[P(no DR), P(any DR)]`, with output shape
`(batch, 2)`. The two output units implement the paper's binary setting; the
paper does not state their count explicitly.

The classifier compiles with Adam and sparse categorical cross-entropy so the
existing scalar `0`/`1` labels from `scripts/dataset.py` can be used directly.
The dataset pipeline already applies DenseNet input preprocessing; do not
apply it again in the classifier. If calculating a binary score such as AUC,
use output column 1 as `P(any DR)`.

```python
import config
from scripts.classifier import build_classifier, set_classifier_backbone_trainable

model = build_classifier()  # pretrained backbone frozen by default
# After the frozen-backbone stage, to fine-tune the final dense block:
set_classifier_backbone_trainable(
    model, 1, learning_rate=config.DENSENET_MINIMUM_LEARNING_RATE
)
```

The fine-tuning call recompiles with a fresh optimizer after changing the
trainable layers. The paper does not give an unfreezing schedule. Its DenseNet
settings—batch size 32, 50 epochs, initial learning rate 0.01, and minimum
learning rate 0.00005—are in `config.py`. Dropout is disabled by default
because the paper does not specify it; it remains configurable for separate
experiments. Sparse categorical cross-entropy is the project's implementation
choice for two softmax outputs and scalar class labels, not a loss specified
by the paper. See `reports/section2b_kaiyao.md` for scope and source alignment.

Previous verification outputs for the single-sigmoid classifier describe
that earlier model. The two-output model has its own record at
`reports/verification/20261016_174508_321406/verification.json`. That
diagnostic did not reproduce a full training run or the paper's reported
performance.

## Section 3a: training baseline (Kaiyao)

From the repository root, after generating the 224 × 224 processed PNGs,
run the training script with its default configuration:

```bash
python -m scripts.train
```

The baseline uses the frozen ImageNet-pretrained DenseNet121 backbone from
Section 2a and the Section 2b global-average-pooling/two-output softmax head.
It trains with Adam at an initial learning rate of 0.01, batch size 32, and
up to 50 epochs. The loss is sparse categorical cross-entropy for the existing
integer labels (`0` = no DR, `1` = any DR). Class weights are calculated from
the **training split only**. The validation split controls callbacks and model
selection; the test split is reserved for later evaluation.

Each ignored `runs/section3a/<UTC timestamp>/` directory contains
`metadata.json`, `history.csv`, the best trainable weights, and full saved
models for the final and best epochs. Frozen backbone features are cached in
memory by default so 50 epochs of head training do not rerun the backbone.
The saved full models support inference; they do not preserve the cached-head
optimizer state for resuming training. The baseline
does not use image augmentation or dropout. The team's existing 70/10/20
train/validation/test split differs from the paper's 80/20 split. The paper
provides the model family, optimizer, batch size, epoch count, and learning
rate bounds, but does not define our class-weight formula, callback schedule,
or sparse loss. The chosen learning-rate callback is ReduceLROnPlateau with
factor 0.5, patience 3, and a 0.00005 floor. See
`reports/section3a_training.md` for the paper-alignment notes and run record.
