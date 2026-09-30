# Kaiyao Hou — Section 1c input pipeline handoff

`scripts/dataset.py` reads the project's existing `data/splits/train.csv`, `val.csv`, and `test.csv`. It resolves processed images through `config.PREPROCESSED_IMAGE_DIR` and expects 224 × 224 RGB PNGs named `{id_code}.png`. It does not create or alter the team split.

## Add to the repository

Copy `scripts/dataset.py` and `tests/test_dataset.py` to those paths in the repository. Add `Pillow` to the repository's existing `requirements.txt` for batch-figure rendering; this handoff includes a revised requirements file for reference. **This README is a handoff note: do not overwrite the repository's root README with it.** Commit the code, tests, Section 1c report, validation JSON files, and batch figure. Keep the roughly 10 GB source archive and 3,662 processed PNGs in local data storage rather than the repository package. Run the following commands from the repository root:

```bash
# Uses the committed CSVs; processed images and TensorFlow are not needed.
python -m scripts.dataset --weights-only

# Requires all processed images referenced by train, val, and test manifests.
python -m scripts.dataset

# Optional figure of raw processed images instead of augmented images.
python -m scripts.dataset --raw-preview

python -m unittest discover -s tests -p test_dataset.py -v
```

The first command writes `reports/class_weights.json`. The full command additionally writes `reports/figures/sample_training_batch.png` from an augmented training batch. `--batch-size N` changes the figure batch size; otherwise it uses `config.DENSENET_BATCH_SIZE` (32). Both figure commands check that every manifest image exists before drawing. The included figure came from the complete processed corpus.

## Use from training code

```python
from scripts.dataset import build_tf_dataset, compute_class_weights, load_splits

splits = load_splits()  # Checks image paths for all three splits by default.
train_ds = build_tf_dataset(splits["train"], training=True)
val_ds = build_tf_dataset(splits["val"], training=False)
test_ds = build_tf_dataset(splits["test"], training=False)
class_weights = compute_class_weights(splits["train"])

# After defining and compiling a compatible binary DenseNet121 model:
# model.fit(train_ds, validation_data=val_ds, class_weight=class_weights)
```

The pipeline casts pixels to `float32`, applies seeded flips and quarter-turn rotations to **training images only**, then calls `tf.keras.applications.densenet.preprocess_input`. Account for that preprocessing when defining the model. Validation and test data are neither shuffled nor augmented. The default seed is `config.RANDOM_SEED` (0).

## Results and verification

The supplied manifests contain 2,562 training images (1,263 no DR; 1,299 any DR), 367 validation images (181; 186), and 733 test images (361; 372), with no ID overlap. Training-label-only balanced weights are **1.01425178147** for no DR and **0.98614318707** for any DR; see `reports/class_weights.json`.

The complete APTOS archive's 3,662 labeled `train_images` matched the supplied `all_splits.csv` IDs and grades. The team's preprocessing script produced all 3,662 processed PNGs with zero skipped and zero failed; every file matched a manifest row and decoded as 224 × 224 RGB. See `reports/full_corpus_validation.json` and the source archive SHA-256 in `reports/figure_sources.json`.

In TensorFlow 2.21, full iteration streamed all 3,662 images through their assigned data pipelines: 81 training batches (2,562 images), 12 validation batches (367), and 23 test batches (733). Every output image was finite `float32` with shape `(224, 224, 3)`; all observed class counts matched the manifests. Each first batch had image shape `(32, 224, 224, 3)` and 32 labels. The first augmented training batch contained 16 no-DR and 16 any-DR images by chance; the figure `reports/figures/sample_training_batch.png` shows this batch, and `reports/batch_validation.json` records the batch and full-iteration checks. The full reconstructed-repository test run passed **23 tests and 2 subtests**. No validation or test labels enter the class-weight calculation, and augmentation is restricted to training. A full model training run was not part of this handoff.
