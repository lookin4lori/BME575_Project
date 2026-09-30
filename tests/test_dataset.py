"""Synthetic, data-free checks for Kaiyao Hou's Section 1c input pipeline.

Run from the repository root with ``python -m unittest discover -s tests -v``.
The CSV fixture uses the columns in the team's committed APTOS split files.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

# The deliverable places this file in tests/ and dataset.py in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import dataset  # noqa: E402


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split_dir = self.root / "data" / "splits"
        self.image_dir = self.root / "data" / "processed" / "aptos2019_224"
        self.split_dir.mkdir(parents=True)
        self.image_dir.mkdir(parents=True)

        # Train is intentionally imbalanced; held-out rows have a different
        # distribution, so fitting weights on all splits would change them.
        self.rows = {
            "train": [("zero_a", 0), ("zero_b", 0), ("zero_c", 0), ("four", 4)],
            "val": [("val_zero", 0), ("val_one", 1)],
            "test": [("test_two", 2), ("test_three", 3)],
        }
        self._write_fixtures()

    def _write_fixtures(self) -> None:
        for split, rows in self.rows.items():
            with (self.split_dir / f"{split}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["id_code", "filename", "diagnosis", "binary_label", "split"]
                )
                for image_id, grade in rows:
                    writer.writerow(
                        [image_id, f"{image_id}.png", grade, int(grade > 0), split]
                    )
            for image_id, grade in rows:
                Image.new("RGB", (224, 224), (30 + grade, 50, 70)).save(
                    self.image_dir / f"{image_id}.png"
                )

    def test_real_manifest_schema_and_binary_mapping(self) -> None:
        splits = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )
        self.assertEqual(
            splits["train"]["binary_label"].tolist(), [0, 0, 0, 1]
        )
        self.assertEqual(splits["val"]["binary_label"].tolist(), [0, 1])
        self.assertEqual(splits["test"]["binary_label"].tolist(), [1, 1])
        self.assertTrue(
            str(splits["train"].iloc[0]["image_path"]).endswith("zero_a.png")
        )

    def test_binary_label_must_agree_with_five_grade_diagnosis(self) -> None:
        path = self.split_dir / "train.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["id_code", "filename", "diagnosis", "binary_label", "split"]
            )
            writer.writerow(["four", "four.png", 4, 0, "train"])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            dataset.read_manifest(path, self.image_dir)

    def test_cross_split_id_overlap_is_rejected(self) -> None:
        self.rows["val"] = [("zero_a", 0)]
        self._write_fixtures()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            dataset.load_splits(split_dir=self.split_dir, image_dir=self.image_dir)

    def test_class_weights_use_training_labels_only(self) -> None:
        splits = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )
        weights = dataset.compute_class_weights(splits["train"])
        self.assertAlmostEqual(weights[0], 2 / 3)
        self.assertAlmostEqual(weights[1], 2.0)
        # With held-out labels included the class proportions differ; this
        # guards against accidentally fitting weights on all three manifests.
        self.assertEqual(len(splits["val"]) + len(splits["test"]), 4)
        self.assertNotEqual(weights[0], 1.0)

    def test_cli_persists_weights_fitted_to_train_only(self) -> None:
        report_dir = self.root / "reports"
        argv = [
            "dataset.py",
            "--split-dir", str(self.split_dir),
            "--image-dir", str(self.image_dir),
            "--output-dir", str(report_dir),
            "--raw-preview",
        ]
        with patch.object(sys, "argv", argv):
            dataset.main()
        result = json.loads((report_dir / "class_weights.json").read_text())
        self.assertEqual(result["train_counts"], {"0": 3, "1": 1})
        self.assertAlmostEqual(result["class_weights"]["0"], 2 / 3)
        self.assertAlmostEqual(result["class_weights"]["1"], 2.0)
        self.assertTrue((report_dir / "figures" / "sample_training_batch.png").is_file())

    def test_training_weights_require_both_classes(self) -> None:
        train = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )["train"]
        with self.assertRaisesRegex(ValueError, "Both classes"):
            dataset.compute_class_weights(train[train["binary_label"] == 0])

    def test_missing_processed_image_is_reported(self) -> None:
        (self.image_dir / "four.png").unlink()
        with self.assertRaises(FileNotFoundError):
            dataset.load_splits(split_dir=self.split_dir, image_dir=self.image_dir)

    def test_augmentation_is_rejected_for_validation_and_test(self) -> None:
        splits = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )
        for name in ("val", "test"):
            with self.subTest(split=name):
                # The guard runs before importing TensorFlow, making this
                # meaningful even on a machine without the training runtime.
                with self.assertRaisesRegex(ValueError, "only for training"):
                    dataset.build_tf_dataset(
                        splits[name], training=False, augment=True
                    )

    def test_tensorflow_train_and_eval_batches_keep_eval_unaugmented(self) -> None:
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            self.skipTest("TensorFlow is not installed")

        # Unequal color channels and spatial gradients reveal a flip, turn,
        # channel swap, or unintended image transformation in eval mode.
        y, x = np.mgrid[0:224, 0:224]
        pixels = np.stack(
            [(x + 20) % 256, (2 * y + 40) % 256, (x + 3 * y + 60) % 256],
            axis=-1,
        ).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(self.image_dir / "val_zero.png")
        Image.fromarray(np.flipud(pixels), mode="RGB").save(
            self.image_dir / "val_one.png"
        )
        splits = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )

        train_batch = next(
            iter(dataset.build_tf_dataset(
                splits["train"], batch_size=4, training=True, seed=17
            ))
        )
        self.assertEqual(tuple(train_batch[0].shape), (4, 224, 224, 3))
        self.assertEqual(tuple(train_batch[1].shape), (4,))
        self.assertEqual(train_batch[0].dtype, tf.float32)
        self.assertEqual(train_batch[1].dtype, tf.float32)

        eval_images, eval_labels = next(
            iter(dataset.build_tf_dataset(
                splits["val"], batch_size=2, training=False, seed=17
            ))
        )
        self.assertEqual(tuple(eval_images.shape), (2, 224, 224, 3))
        self.assertEqual(eval_images.dtype, tf.float32)
        np.testing.assert_array_equal(eval_labels.numpy(), [0.0, 1.0])
        expected = tf.keras.applications.densenet.preprocess_input(
            np.stack((pixels, np.flipud(pixels))).astype(np.float32)
        )
        np.testing.assert_allclose(eval_images.numpy(), expected, rtol=0, atol=1e-6)

    def test_raw_training_batch_figure_uses_processed_pngs(self) -> None:
        train = dataset.load_splits(
            split_dir=self.split_dir, image_dir=self.image_dir
        )["train"]
        output = dataset.save_sample_batch_figure(
            train, self.root / "sample_batch.png", batch_size=4, augmented=False
        )
        with Image.open(output) as figure:
            self.assertGreater(figure.width, 224)
            self.assertGreater(figure.height, 224)


if __name__ == "__main__":
    unittest.main()
