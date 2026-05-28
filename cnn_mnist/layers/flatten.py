"""Flatten layer."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cnn_mnist.layers.base import LayerBase


class Flatten(LayerBase):
    """Flatten non-batch dimensions into one feature dimension."""

    def __init__(self) -> None:
        self._shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Reshape ``(N, ...)`` to ``(N, prod(...))``."""
        del training
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Restore the cached input shape."""
        if self._shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return dout.reshape(self._shape)
