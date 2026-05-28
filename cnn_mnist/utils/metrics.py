"""Classification metrics."""

from __future__ import annotations

from typing import Union

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return scalar classification accuracy."""
    return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Return a ``num_classes x num_classes`` confusion matrix."""
    true: np.ndarray = y_true.astype(np.int64, copy=False)
    pred: np.ndarray = y_pred.astype(np.int64, copy=False)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    out_of_range = (
        np.any(true < 0)
        or np.any(pred < 0)
        or np.any(true >= num_classes)
        or np.any(pred >= num_classes)
    )
    if out_of_range:
        raise ValueError("labels are outside the configured class range")
    flat = true * num_classes + pred
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> dict[str, Union[np.ndarray, float]]:
    """Return per-class precision/recall/F1 and macro F1."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        tp = matrix[cls, cls]
        fp = matrix[:, cls].sum() - tp
        fn = matrix[cls, :].sum() - tp
        precision[cls] = tp / (tp + fp) if tp + fp else 0.0
        recall[cls] = tp / (tp + fn) if tp + fn else 0.0
        denom = precision[cls] + recall[cls]
        f1[cls] = 2.0 * precision[cls] * recall[cls] / denom if denom else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": matrix,
    }
