"""Centered-difference gradient checking utilities."""

from __future__ import annotations

import numpy as np

from cnn_mnist.layers.base import LayerBase


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Return the maximum elementwise relative error."""
    denom = np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-8)
    return float(np.max(np.abs(analytic - numeric) / denom))


def check_layer_gradient(
    layer: LayerBase,
    x: np.ndarray,
    dout: np.ndarray,
    h: float = 1e-5,
) -> dict[str, float]:
    """Compare analytical parameter gradients with centered-difference gradients."""
    layer.forward(x, training=True)
    layer.backward(dout)
    errors: dict[str, float] = {}
    for name, param in layer.params().items():
        analytic = layer.grads()[name].copy()
        numeric = np.zeros_like(param)
        iterator = np.nditer(param, flags=["multi_index"], op_flags=[["readwrite"]])  # type: ignore[arg-type]
        while not iterator.finished:
            index = iterator.multi_index
            original = param[index]
            param[index] = original + h
            plus: float = float(np.sum(layer.forward(x, training=True) * dout))
            param[index] = original - h
            minus: float = float(np.sum(layer.forward(x, training=True) * dout))
            param[index] = original
            numeric[index] = (plus - minus) / (2.0 * h)
            iterator.iternext()
        layer.forward(x, training=True)
        layer.backward(dout)
        errors[name] = relative_error(analytic, numeric)
    return errors
