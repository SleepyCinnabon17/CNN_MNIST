"""Activation layers."""

from __future__ import annotations

import numpy as np
from typing import Optional

from cnn_mnist.layers.base import LayerBase


class ReLU(LayerBase):
    """Rectified linear activation."""

    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Return ``max(0, x)``."""
        del training
        self._mask = x > 0
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Pass gradients only where the cached input was positive."""
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return dout * self._mask


class Softmax(LayerBase):
    """Numerically stable standalone softmax activation.

    Training models in this project pass logits directly to
    ``SoftmaxCrossEntropyLoss`` for the combined stable gradient; this layer is
    provided for API completeness, inference-style probabilities, and explicit
    Jacobian-vector product testing.
    """

    def __init__(self) -> None:
        self._out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Return per-row probabilities that sum to one."""
        del training
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._out = exp / np.sum(exp, axis=1, keepdims=True)
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Return the Jacobian-vector product for each sample."""
        if self._out is None:
            raise RuntimeError("Softmax.backward called before forward")
        dot = np.sum(dout * self._out, axis=1, keepdims=True)
        return self._out * (dout - dot)
