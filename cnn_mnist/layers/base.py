"""Common layer interface."""

from __future__ import annotations

from typing import Dict

import numpy as np


ArrayDict = Dict[str, np.ndarray]


class LayerBase:
    """Base interface for forward/backward layers."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Return the layer output for ``x``."""
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Return the gradient with respect to this layer's input."""
        raise NotImplementedError

    def params(self) -> ArrayDict:
        """Return mutable parameter arrays keyed by parameter name."""
        return {}

    def grads(self) -> ArrayDict:
        """Return gradient arrays keyed like ``params``."""
        return {}
