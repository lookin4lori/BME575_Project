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

`scripts/classifier.py` builds the DenseNet121 classifier on the existing
`scripts/model.py` backbone: global average pooling, configurable dropout,
and one sigmoid output for no DR (0) versus any DR (1). It compiles with Adam
and binary cross-entropy. The input should come from `scripts/dataset.py`,
which already applies DenseNet preprocessing; do not preprocess it again.

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
trainable layers. Using the configured minimum rate for that stage is an
example choice; the paper does not give an unfreezing schedule. The paper's
DenseNet settings are batch size 32, 50 epochs,
initial learning rate 0.01, and minimum learning rate 0.00005; these are in
`config.py`. The 0.5 dropout rate and single-sigmoid/binary-cross-entropy
formulation are project choices for this binary-label pipeline, not settings
specified by the paper. See `reports/section2b_kaiyao.md` for scope and
source alignment.
