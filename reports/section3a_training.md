# Section 3a — DenseNet121 training baseline (Kaiyao Hou)

Reference: [Mohanty et al., *Sensors* 2023, 23, 5726](https://www.mdpi.com/1424-8220/23/12/5726).

## Purpose and run command

Run the existing APTOS 2019 binary classification pipeline from the repository
root after Section 1 preprocessing has produced 224 × 224 RGB PNGs:

```bash
python -m scripts.train
```

The script reads the committed split manifests and local processed images.
Training artifacts belong in a separate per-run directory under
`runs/section3a/<UTC timestamp>/`; Git ignores that directory and the raw or
processed images. `--no-cache-features` can be used to recompute backbone
features during training, and `--output-dir` chooses another output directory.
The script also exposes `--augment`, `--train-limit`, `--val-limit`, and
`--weights none` for separate diagnostics; these are not the default baseline.
Use `--augment` together with `--no-cache-features`, since a fixed feature
cache cannot represent fresh per-epoch augmentation. The validation data may
be used to select a checkpoint and adjust learning rate. Test split counts and
manifest hashes are recorded for integrity, but test images and labels are
not used to fit, monitor, or select the model.

## Paper alignment and implementation choices

| Item | Baseline | Basis |
| --- | --- | --- |
| Architecture | ImageNet-pretrained DenseNet121 with frozen feature extractor, global average pooling, two-class softmax | Paper §4.3 describes DenseNet121, global average pooling, fully connected softmax classification; the repository's Section 2a/2b implementation chooses ImageNet weights, freezing, and two output units. |
| Input and target | Preprocessed 224 × 224 RGB; scalar `0` (no DR) or `1` (any DR) | Paper §4.1 specifies 224 × 224 preparation; the repository's binary label convention is in `config.py` and `scripts/dataset.py`. |
| Optimizer | Adam with initial learning rate 0.01 | Paper §4.4 and Table 4. |
| Batch and duration | Batch size 32, 50 scheduled epochs | Paper Table 4. |
| Learning-rate floor | 0.00005 | Paper Table 4. The paper does not give a complete schedule; this baseline uses ReduceLROnPlateau on validation loss, factor 0.5, patience 3, with that floor. |
| Loss and weights | Sparse categorical cross-entropy; balanced class weights `n_train / (2 × n_class)` | Project implementation for two softmax outputs and existing scalar labels. Weight counts come only from the training split. The paper does not specify this loss or weighting formula. |
| Regularization | No dropout and no training augmentation in the baseline | Paper does not specify dropout or an augmentation policy for this setting. Both are separate experiments if enabled later. |
| Data split | Existing stratified 70% train, 10% validation, 20% test | Team split in `config.py`; differs from the paper's 80/20 division. Results must not be described as exact reproduction of the paper's data protocol. |

The Section 2b head returns `[P(no DR), P(any DR)]`. Sparse cross-entropy
accepts the current `0`/`1` labels without one-hot encoding. DenseNet input
normalization happens in `scripts/dataset.py` and must not be applied again
inside the classifier. The frozen backbone features may be cached locally so
the 50-epoch head run does not recompute the same image features. That cache
is only valid for the fixed preprocessing and frozen backbone.

## Validation and saved evidence

The default training run monitors validation loss. It writes `history.csv`
with per-epoch metrics and learning rate, `best_trainable.weights.h5` with
weights from the epoch of lowest validation loss, `final_model.keras` with the
last-epoch image model, and `best_model.keras` with the selected full image
model. `metadata.json` records configuration, split sizes, training-only class
counts and weights, hashes of the split manifests, and best epoch metrics.
The `.keras` image models are inference artifacts. Their files do not preserve
the cached-head optimizer state for resuming that training path; a resumed
training run must create and compile a fresh optimizer.
The callback schedule and selected checkpoint are implementation details of
this run, not parameters supplied by the paper. No test-set metric belongs in
the training or checkpoint-selection record.

## Run results

The full default baseline completed on 2026-10-19 in
`runs/section3a/20261019T213748_201239Z/` (UTC run ID). It used all 2,562
training images, all 367 validation images, and all 50 scheduled epochs.
The 733-image test split was held out from fitting, validation monitoring, and
checkpoint selection. The run used a frozen ImageNet DenseNet121, cached
features, no augmentation or dropout, Adam starting at 0.01, batch size 32,
and the validation-loss learning-rate callback described above. The final
epoch's logged learning rate was 0.00005.

Training counts were 1,263 no DR and 1,299 any DR, yielding balanced class
weights **1.0142517815** and **0.9861431871**, respectively. Validation
counts were 181 no DR and 186 any DR. The best checkpoint by validation loss
was at **epoch 24**: validation loss **0.0793581** and sparse categorical
accuracy **0.9700273** (97.00%). At epoch 50, validation loss was **0.0795099**
and accuracy **0.9645777** (96.46%). These are validation results on this
team's split, not held-out test metrics or a reproduction of the paper's
reported accuracy.

The committed evidence includes the [run metadata](training/20261019T213748_201239Z/metadata.json)
and [50-epoch CSV history](training/20261019T213748_201239Z/history.csv).
The local run directory also contains the best and final model artifacts
described above. Model files and images remain local; they are not committed
to Git. A separate evaluation on the held-out test split is still required
before reporting test performance.
