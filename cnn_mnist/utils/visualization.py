"""Matplotlib artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(history: dict[str, list[float]], output_dir: Union[str, Path]) -> None:
    """Save loss and accuracy curves from training history."""
    plt = _plt()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.get("train_loss", []), label="train")
    ax.plot(history.get("val_loss", []), label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.get("train_accuracy", []), label="train")
    ax.plot(history.get("val_accuracy", []), label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_dir: Union[str, Path]) -> None:
    """Save a confusion-matrix heatmap."""
    plt = _plt()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Union[str, Path],
    count: int = 16,
) -> None:
    """Save a grid of sample predictions."""
    plt = _plt()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = min(count, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for ax, index in zip(axes.ravel(), range(count)):
        ax.imshow(images[index, 0], cmap="gray")
        ax.set_title(f"t={int(y_true[index])} p={int(y_pred[index])}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "sample_predictions.png", dpi=150)
    plt.close(fig)


def plot_conv_filters(filters: np.ndarray, output_dir: Union[str, Path]) -> None:
    """Save layer-1 convolution filters as a grid."""
    plt = _plt()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_filters = filters.shape[0]
    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, index in zip(axes_arr, range(num_filters)):
        filt = filters[index, 0]
        ax.imshow(filt, cmap="gray")
        ax.axis("off")
    for ax in axes_arr[num_filters:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "conv1_filters.png", dpi=150)
    plt.close(fig)
