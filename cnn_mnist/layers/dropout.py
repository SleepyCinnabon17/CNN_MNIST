"""Inverted dropout layer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from cnn_mnist.layers.base import LayerBase


class Dropout(LayerBase):
    """Randomly zero activations during training and disable during evaluation."""

    def __init__(self, rate: float, rng: Optional[np.random.Generator] = None) -> None:
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rate must be in [0, 1)")
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.rng = rng or np.random.default_rng()
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply inverted dropout in training mode."""
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) < self.keep_prob) / self.keep_prob
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Apply the same dropout mask to upstream gradients."""
        if self._mask is None:
            return dout
        return dout * self._mask
