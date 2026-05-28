"""Batch normalization layer for dense or convolutional activations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cnn_mnist.layers.base import ArrayDict, LayerBase


class BatchNorm(LayerBase):
    """Batch normalization with running statistics."""

    def __init__(
        self,
        num_features: int,
        rng: Optional[np.random.Generator] = None,
        momentum: float = 0.9,
        eps: float = 1e-5,
    ) -> None:
        # Accepted for API consistency with layers that require random initialization.
        del rng
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self._shape: Optional[Tuple[int, ...]] = None
        self._x_hat: Optional[np.ndarray] = None
        self._x_centered: Optional[np.ndarray] = None
        self._inv_std: Optional[np.ndarray] = None

    def _to_matrix(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x
        if x.ndim == 4:
            return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])
        raise ValueError("BatchNorm supports 2D or 4D inputs")

    def _from_matrix(self, x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) == 2:
            return x.reshape(shape)
        n, c, h, w = shape
        return x.reshape(n, h, w, c).transpose(0, 3, 1, 2)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Normalize activations using batch stats in training and running stats in eval."""
        self._shape = x.shape
        matrix = self._to_matrix(x)
        if training:
            mean = matrix.mean(axis=0)
            var = matrix.var(axis=0)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        centered = matrix - mean
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = centered * inv_std
        out = x_hat * self.gamma + self.beta
        if training:
            self._x_hat = x_hat
            self._x_centered = centered
            self._inv_std = inv_std
        return self._from_matrix(out, x.shape)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagate through batch normalization."""
        if self._shape is None or self._x_hat is None:
            raise RuntimeError("BatchNorm.backward called before training forward")
        if self._x_centered is None or self._inv_std is None:
            raise RuntimeError("BatchNorm cache is incomplete")
        dout_matrix = self._to_matrix(dout)
        m = dout_matrix.shape[0]
        self.dbeta = dout_matrix.sum(axis=0)
        self.dgamma = np.sum(dout_matrix * self._x_hat, axis=0)
        dx_hat = dout_matrix * self.gamma
        dvar = np.sum(dx_hat * self._x_centered * -0.5 * self._inv_std**3, axis=0)
        dmean = np.sum(dx_hat * -self._inv_std, axis=0)
        dmean += dvar * np.sum(-2.0 * self._x_centered, axis=0)
        dx = dx_hat * self._inv_std
        dx += dvar * 2.0 * self._x_centered / m
        dx += dmean / m
        return self._from_matrix(dx, self._shape)

    def params(self) -> ArrayDict:
        """Return learnable scale and offset."""
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self) -> ArrayDict:
        """Return gradients for scale and offset."""
        return {"gamma": self.dgamma, "beta": self.dbeta}
