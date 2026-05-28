"""Max-pooling layer."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from cnn_mnist.layers.base import LayerBase


class MaxPool2D(LayerBase):
    """2D max pooling with argmax routing for backpropagation."""

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = 2, stride: int = 2) -> None:
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size
        self.stride = stride
        self._x: Optional[np.ndarray] = None
        self._argmax: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Downsample each channel by taking window maxima."""
        del training
        x = np.ascontiguousarray(x)
        n, c, h, w = x.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore[call-overload]
            x,
            (ph, pw),
            axis=(2, 3),
        )
        windows = windows[
            :,
            :,
            : out_h * self.stride : self.stride,
            : out_w * self.stride : self.stride,
            :,
            :,
        ]
        window_flat = windows.reshape(n, c, out_h, out_w, ph * pw)
        argmax = np.argmax(window_flat, axis=4)
        out = np.max(window_flat, axis=4)
        self._x = x
        self._argmax = argmax
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Route upstream gradients to the cached argmax positions."""
        if self._x is None or self._argmax is None:
            raise RuntimeError("MaxPool2D.backward called before forward")
        n, c, out_h, out_w = dout.shape
        ph, pw = self.pool_size
        dx = np.zeros_like(self._x)
        if self.stride == ph == pw:
            n_idx = np.arange(n)[:, None, None, None]
            c_idx = np.arange(c)[None, :, None, None]
            h_idx = np.arange(out_h)[None, None, :, None] * self.stride
            w_idx = np.arange(out_w)[None, None, None, :] * self.stride
            flat = self._argmax
            p_pos: np.ndarray = flat // pw
            q_pos: np.ndarray = flat % pw
            dx[n_idx, c_idx, h_idx + p_pos, w_idx + q_pos] = dout
            return dx
        n_idx = np.arange(n)[:, None]
        c_idx = np.arange(c)[None, :]
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                flat = self._argmax[:, :, i, j]
                p_pos = flat // pw
                q_pos = flat % pw
                np.add.at(dx, (n_idx, c_idx, hs + p_pos, ws + q_pos), dout[:, :, i, j])
        return dx
