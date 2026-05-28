"""Vectorized 2D convolution implemented with NumPy im2col/col2im."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from cnn_mnist.layers.base import ArrayDict, LayerBase


class Conv2D(LayerBase):
    """Cross-correlation layer for inputs in ``(N, C, H, W)`` format."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: str = "valid",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if padding not in {"same", "valid"}:
            raise ValueError("padding must be 'same' or 'valid'")
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rng = rng or np.random.default_rng()
        kh, kw = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kh * kw))
        self.W = self.rng.normal(0.0, scale, size=(num_filters, in_channels, kh, kw))
        self.b = np.zeros(num_filters, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cols: Optional[np.ndarray] = None
        self._x_shape: Optional[Tuple[int, int, int, int]] = None
        self._pad: Optional[Tuple[int, int, int, int]] = None
        self._out_hw: Optional[Tuple[int, int]] = None

    def _padding(self, h: int, w: int) -> tuple[int, int, int, int, int, int]:
        kh, kw = self.kernel_size
        if self.padding == "same":
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
            pad_h = max((out_h - 1) * self.stride + kh - h, 0)
            pad_w = max((out_w - 1) * self.stride + kw - w, 0)
        else:
            out_h = (h - kh) // self.stride + 1
            out_w = (w - kw) // self.stride + 1
            pad_h = 0
            pad_w = 0
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return top, bottom, left, right, out_h, out_w

    def _im2col(self, x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        n, c, _, _ = x.shape
        kh, kw = self.kernel_size
        cols = np.empty((n, out_h, out_w, c, kh, kw), dtype=x.dtype)
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                cols[:, i, j] = x[:, :, hs : hs + kh, ws : ws + kw]
        return cols.reshape(n * out_h * out_w, c * kh * kw)

    def _col2im(
        self,
        cols: np.ndarray,
        padded_shape: tuple[int, int, int, int],
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        n, c, _, _ = padded_shape
        kh, kw = self.kernel_size
        dx = np.zeros(padded_shape, dtype=cols.dtype)
        col_view = cols.reshape(n, out_h, out_w, c, kh, kw)
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                dx[:, :, hs : hs + kh, ws : ws + kw] += col_view[:, i, j]
        return dx

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply filters to the input volume."""
        del training
        n, _, h, w = x.shape
        top, bottom, left, right, out_h, out_w = self._padding(h, w)
        x_padded = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
        cols = self._im2col(x_padded, out_h, out_w)
        w_flat = self.W.reshape(self.num_filters, -1)
        out = cols @ w_flat.T + self.b
        self._cols = cols
        self._x_shape = x.shape
        self._pad = (top, bottom, left, right)
        self._out_hw = (out_h, out_w)
        return out.reshape(n, out_h, out_w, self.num_filters).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Compute gradients for filters, bias, and input."""
        if self._cols is None or self._x_shape is None or self._pad is None or self._out_hw is None:
            raise RuntimeError("Conv2D.backward called before forward")
        n, _, h, w = self._x_shape
        top, bottom, left, right = self._pad
        out_h, out_w = self._out_hw
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)
        self.dW = (dout_flat.T @ self._cols).reshape(self.W.shape)
        self.db = dout_flat.sum(axis=0)
        dcols = dout_flat @ self.W.reshape(self.num_filters, -1)
        dx_padded = self._col2im(dcols, (n, self.in_channels, h + top + bottom, w + left + right), out_h, out_w)
        if top == bottom == left == right == 0:
            return dx_padded
        h_stop = -bottom if bottom else None
        w_stop = -right if right else None
        return dx_padded[:, :, top:h_stop, left:w_stop]

    def params(self) -> ArrayDict:
        """Return convolution parameters."""
        return {"W": self.W, "b": self.b}

    def grads(self) -> ArrayDict:
        """Return convolution parameter gradients."""
        return {"W": self.dW, "b": self.db}
