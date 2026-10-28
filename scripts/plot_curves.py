"""Plot training and validation learning curves from a train.py history.csv (anais- Section 3c).

This readsany history file train.py makes, so I used this exact same script for comparison model Inception V3

Have to run from the repo root, for example:
    python -m scripts.plot_curves reports/training/20261019T213748_201239Z/history.csv --label DenseNet121
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import pandas as pd

import config

ACC = "sparse_categorical_accuracy"


def load_history(path):
    """Read a history.csv and shift epochs to start at 1 (train.py counts from 0)."""
    history = pd.read_csv(path)
    required = {"epoch", "loss", "val_loss", ACC, f"val_{ACC}"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    history["epoch"] = history["epoch"] + 1
    return history


def plot_curves(history, label, output_path):
    """Save side-by-side loss and accuracy curves, marking the best validation-loss epoch."""
    best = history.loc[history["val_loss"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [(axes[0], "loss", "val_loss", "Loss"),
              (axes[1], ACC, f"val_{ACC}", "Accuracy")]
    for ax, train_col, val_col, title in panels:
        ax.plot(history["epoch"], history[train_col], label="Training")
        ax.plot(history["epoch"], history[val_col], label="Validation")
        ax.axvline(best["epoch"], color="gray", linestyle="--",
                   label=f"Best epoch ({int(best['epoch'])})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{label}: {title}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return best


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from a history.csv.")
    parser.add_argument("history", type=Path, help="Path to a train.py history.csv")
    parser.add_argument("--label", default="DenseNet121", help="Model name for titles and file name")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the figure")
    args = parser.parse_args()

    output = args.output or (config.REPORT_DIR / "figures" /
                             f"{args.label.lower().replace(' ', '_')}_learning_curves.png")
    best = plot_curves(load_history(args.history), args.label, output)
    print(f"Best epoch (1-based): {int(best['epoch'])}  "
          f"val_loss={best['val_loss']:.4f}  val_acc={best[f'val_{ACC}']:.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()