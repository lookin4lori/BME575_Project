import hashlib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scripts import evaluate


def example_predictions(): #four made up images jsut to test
    probabilities = np.array([
        [0.9, 0.1],  # True 0, predicted 0: correct
        [0.4, 0.6],  # True 0, predicted 1: false positive
        [0.6, 0.4],  # True 1, predicted 0: false negative
        [0.1, 0.9],  # True 1, predicted 1: correct
    ], dtype=np.float32)

    frame = pd.DataFrame({
        "id_code": ["image_b", "image_a", "image_d", "image_c"],
        "diagnosis": [0, 0, 2, 4],
        "true_label": [0, 0, 1, 1],
        "probability_no_dr": probabilities[:, 0],
        "probability_dr": probabilities[:, 1],
        "predicted_label": probabilities.argmax(axis=1),
    })

    frame["correct"] = (
        frame["true_label"] == frame["predicted_label"]
    )
    return frame


# 1. Check metrics against answers we already know.
def test_known_metric_values():
    metrics = evaluate.calculate_metrics(example_predictions())
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity_dr"] == pytest.approx(0.5)
    assert metrics["specificity_no_dr"] == pytest.approx(0.5)
    assert metrics["roc_auc_dr"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


# check confusion-matrix rows/columns are not reversed
def test_confusion_matrix_orientation():
    frame = example_predictions()
    # predict every image as No DR
    frame["probability_no_dr"] = 0.9
    frame["probability_dr"] = 0.1
    frame["predicted_label"] = 0
    metrics = evaluate.calculate_metrics(frame)

    # rows = actual class; columns = predicted class.
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]


# cCheck image order DR probability column, and eval mode
def test_prediction_order_and_evaluation_mode(monkeypatch):
    frame = pd.DataFrame({
        "id_code": ["image_b", "image_a", "image_d", "image_c"],
        "diagnosis": [0, 0, 2, 4],
        "binary_label": [0, 0, 1, 1],
    })

    probabilities = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.1, 0.9],
    ], dtype=np.float32)

    def fake_dataset(received_frame, **options):
        assert received_frame["id_code"].tolist() == frame["id_code"].tolist()
        assert options["training"] is False
        assert options["augment"] is False

        return tf.data.Dataset.from_tensor_slices((
            probabilities,
            frame["binary_label"].to_numpy(dtype=np.float32),
        )).batch(2)

    class FakeModel:
        def __call__(self, images, training=False):
            assert training is False
            return images

    # temp sub fake data loader for this test only.
    monkeypatch.setattr(evaluate, "build_tf_dataset", fake_dataset)

    result = evaluate.predict_frame(FakeModel(), frame, batch_size=2)

    assert result["id_code"].tolist() == frame["id_code"].tolist()
    assert result["predicted_label"].tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(
        result["probability_dr"], probabilities[:, 1]
    )


# cehck counts and correct predictions within grades.
def test_subgroup_counts():
    groups = evaluate.make_subgroups(
        example_predictions()
    ).set_index("original_grade")

    assert groups["number_of_images"].sum() == 4
    assert groups.loc[0, "number_of_images"] == 2
    assert groups.loc[0, "correct"] == 1
    assert groups.loc[2, "correct"] == 0
    assert groups.loc[4, "correct"] == 1
    assert groups.loc[1, "number_of_images"] == 0


# allow diff line endings but reject changed data.
def test_manifest_line_endings_and_changed_content(tmp_path):
    path = tmp_path / "val.csv"
    lf = b"id_code,diagnosis\nexample,0\n"
    windows = lf.replace(b"\n", b"\r\n")
    expected_hash = hashlib.sha256(windows).hexdigest()

    path.write_bytes(lf)
    assert evaluate.check_manifest(path, expected_hash) == "CRLF"

    path.write_bytes(windows)
    assert evaluate.check_manifest(path, expected_hash) == "exact"

    # changing the label must fail the integrity check... for safetty
    path.write_bytes(b"id_code,diagnosis\nexample,1\n")
    with pytest.raises(ValueError):
        evaluate.check_manifest(path, expected_hash)