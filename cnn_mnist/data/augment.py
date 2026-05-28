"""Training-time image augmentation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_shift(
    images: np.ndarray,
    max_shift: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply zero-padded random horizontal and vertical shifts."""
    rng = rng or np.random.default_rng()
    if max_shift <= 0:
        return images.copy()
    shifted = np.zeros_like(images)
    for index in range(images.shape[0]):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        src_y0 = max(0, -dy)
        src_y1 = images.shape[2] - max(0, dy)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        src_x0 = max(0, -dx)
        src_x1 = images.shape[3] - max(0, dx)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        shifted[index, :, dst_y0:dst_y1, dst_x0:dst_x1] = images[
            index,
            :,
            src_y0:src_y1,
            src_x0:src_x1,
        ]
    return shifted
