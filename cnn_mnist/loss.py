"""Loss functions."""

from __future__ import annotations

import numpy as np
from typing import Optional


class SoftmaxCrossEntropyLoss:
    """Stable softmax cross-entropy over integer or one-hot labels.

    ``eps`` is retained for constructor compatibility. The implementation uses stable log-softmax
    algebra instead of ``log(prob + eps)``, so the standard combined gradient exactly matches the
    forward loss for finite logits.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._probs: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Return the mean cross-entropy loss for a batch."""
        if y_true.ndim == 1:
            targets = np.eye(logits.shape[1], dtype=logits.dtype)[y_true.astype(int)]
        else:
            targets = y_true.astype(logits.dtype)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(exp, axis=1, keepdims=True))
        self._probs = probs
        self._targets = targets
        return float(-np.sum(targets * log_probs) / logits.shape[0])

    def backward(self) -> np.ndarray:
        """Return ``dL/dlogits`` using the combined stable gradient."""
        if self._probs is None or self._targets is None:
            raise RuntimeError("SoftmaxCrossEntropyLoss.backward called before forward")
        return (self._probs - self._targets) / self._probs.shape[0]
