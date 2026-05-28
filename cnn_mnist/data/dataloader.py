"""Mini-batch data loader."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from cnn_mnist.data.augment import random_shift


class DataLoader:
    """Create shuffled mini-batches from NumPy arrays."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        augment: bool = False,
        max_shift: int = 2,
    ) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.max_shift = max_shift
        self.last_indices = np.arange(len(x))

    def batches(self, training: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield mini-batches; augmentation is applied only in training mode."""
        indices = np.arange(len(self.x))
        if self.shuffle:
            self.rng.shuffle(indices)
        self.last_indices = indices.copy()
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_x = self.x[batch_indices]
            if training and self.augment:
                batch_x = random_shift(batch_x, max_shift=self.max_shift, rng=self.rng)
            yield batch_x, self.y[batch_indices]
