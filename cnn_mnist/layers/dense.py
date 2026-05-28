"""Fully connected affine layer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from cnn_mnist.layers.base import ArrayDict, LayerBase


class Dense(LayerBase):
    """Affine layer computing ``x @ W + b``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / input_dim)
        self.W = self.rng.normal(0.0, scale, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the affine output."""
        del training
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Compute gradients for parameters and input."""
        if self._x is None:
            raise RuntimeError("Dense.backward called before forward")
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> ArrayDict:
        """Return layer parameters."""
        return {"W": self.W, "b": self.b}

    def grads(self) -> ArrayDict:
        """Return layer parameter gradients."""
        return {"W": self.dW, "b": self.db}
