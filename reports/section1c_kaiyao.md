### 1c. Model input pipeline — Kaiyao Hou

I implemented the model input pipeline in `scripts/dataset.py` using the team's existing `train.csv`, `val.csv`, and `test.csv`; no new split is created. The loader resolves each `id_code` to a processed RGB PNG, checks the manifest schema and label consistency, and rejects duplicate or overlapping IDs across splits. The binary target is no DR for diagnosis grade 0 and any DR for grades 1–4. The expected processed image is 224 × 224 × 3, with `uint8` pixels before model preprocessing. (*ML_ProgressReport1.pdf*, pp. 5, 8; *HandoverNotes.pdf*, pp. 5, 7, 9–10.)

The TensorFlow `tf.data` pipeline decodes PNGs, checks image shape, casts pixels to `float32`, applies `tf.keras.applications.densenet.preprocess_input`, batches, and prefetches. It shuffles training images and applies seeded random horizontal and vertical flips and 0°, 90°, 180°, or 270° rotations **only to training images**. Validation and test images remain in manifest order without augmentation; the function rejects an attempt to augment them. The default seed is 0 and default batch size is 32, as specified in `config.py`. These augmentation choices implement the handover's train-only requirement. (*ML_ProgressReport1.pdf*, p. 5; *HandoverNotes.pdf*, pp. 7, 9–10.)

The supplied split manifests contain 3,662 distinct IDs and no overlap among splits:

| Split | No DR (0) | Any DR (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 1,263 | 1,299 | 2,562 |
| Validation | 181 | 186 | 367 |
| Test | 361 | 372 | 733 |

To avoid using held-out labels when fitting loss weights, `compute_class_weights` uses **only the 2,562 training labels**. With $w_c=N_{\mathrm{train}}/(2n_c)$, the resulting weights are $w_0=1.01425178147$ and $w_1=0.98614318707$. The values and split counts are saved in `reports/class_weights.json`. The disjoint split IDs, training-only weight calculation, and training-only augmentation prevent leakage through these input-pipeline steps; model training is outside this section.

I verified the complete [APTOS 2019 labeled-image archive](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) against the supplied manifests: all 3,662 archive `train_images` IDs and diagnosis grades matched `all_splits.csv`. Running the team's `scripts/preprocess.py` with its configured settings and `--overwrite` produced 3,662 processed images, with zero skipped or failed. A full-corpus check found exactly one matching, decodable 224 × 224 RGB PNG for every manifest row and no extra files (`reports/full_corpus_validation.json`).

In TensorFlow 2.21, I iterated through **all 3,662 records** in the training, validation, and test pipelines: 81, 12, and 23 batches respectively. Every output image was finite `float32` with shape `(224, 224, 3)`, and the observed class counts matched the manifests. The first batch of each split had image shape `(32, 224, 224, 3)` and label shape `(32,)`. The training data were shuffled and augmented; validation and test data were neither. The first training batch happened to contain 16 no-DR and 16 any-DR images; this batch composition was not imposed on the data. `reports/figures/sample_training_batch.png` shows that augmented training batch after reversing DenseNet normalization solely for display. The batch observations are in `reports/batch_validation.json`; archive and figure hashes are in `reports/figure_sources.json`. The complete reconstructed-repository test run passed **23 tests and 2 subtests**. These checks validate the input pipeline and processed data; no model performance is claimed.
