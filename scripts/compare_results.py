"""Combine held-out test results for DenseNet121 and the comparison model Inception V3 (anais - Section 3c).

Run from the repo root: python -m scripts.compare_results
"""

import json

import pandas as pd

import config

EVAL_DIR = config.REPORT_DIR / "evaluation"

# Our runs: same split, training settings, and frozen ImageNet backbone.
OUR_RUNS = {
    "DenseNet121": EVAL_DIR / "20261023T165435_105761Z_test",
    "Inception V3": EVAL_DIR / "20261027T233953_864315Z_test",
}

# Accuracies from the target paper's Table 5 (APTOS). Other studies used their own protocols.
PAPER_ROWS = [
    ("DenseNet121", "Target paper (Mohanty et al., 2023)", 97.30),
    ("VGG16 + XGBoost (hybrid)", "Target paper (Mohanty et al., 2023)", 79.50),
    ("GoogleNet", "Ref. [72] cited in target paper", 97.00),
    ("CNN", "Ref. [70] cited in target paper", 94.44),
    ("Inception-ResNet-V2", "Ref. [71] cited in target paper", 82.18),
    ("Inception V3", "Ref. [69] cited in target paper", 82.00),
    ("CNN", "Ref. [68] cited in target paper", 75.61),
]


def load_metrics(folder):
    """Read one evaluation folder's metrics.json and return the values for the table."""
    with open(folder / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    report = m["classification_report"]
    return {
        "test_images": m["number_of_images"],
        "accuracy_pct": round(100 * m["accuracy"], 2),
        "sensitivity_dr_pct": round(100 * m["sensitivity_dr"], 2),
        "specificity_no_dr_pct": round(100 * m["specificity_no_dr"], 2),
        "precision_dr_pct": round(100 * report["Any DR"]["precision"], 2),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "roc_auc_dr": round(m["roc_auc_dr"], 4),
    }


def main():
    # Table 1: our models, all metrics, same split and protocol.
    ours = pd.DataFrame([
        {"model": name, **load_metrics(path)} for name, path in OUR_RUNS.items()
    ])
    ours_path = config.REPORT_DIR / "comparison_our_models.csv"
    ours.to_csv(ours_path, index=False)

    # Table 2: accuracy only, our models alongside the paper's Table 5 (highest first).
    literature = pd.concat([
        pd.DataFrame({"model": ours["model"], "source": "This project",
                      "accuracy_pct": ours["accuracy_pct"]}),
        pd.DataFrame(PAPER_ROWS, columns=["model", "source", "accuracy_pct"]),
    ], ignore_index=True).sort_values("accuracy_pct", ascending=False)
    lit_path = config.REPORT_DIR / "comparison_literature.csv"
    literature.to_csv(lit_path, index=False)

    print("Our models (held-out test set):")
    print(ours.to_string(index=False))
    print("\nAccuracy compared with the paper's Table 5:")
    print(literature.to_string(index=False))
    print(f"\nSaved: {ours_path}\nSaved: {lit_path}")

if __name__ == "__main__":
    main()