"""Model persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from cnn_mnist.model import Model


def save_model(model: Model, path: Union[str, Path]) -> None:
    """Save all model parameters to a NumPy ``.npz`` archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **model.state_dict())  # type: ignore[arg-type]


def load_model(model: Model, path: Union[str, Path]) -> None:
    """Load model parameters from a NumPy ``.npz`` archive."""
    with np.load(Path(path), allow_pickle=False) as data:
        model.load_state_dict({key: data[key] for key in data.files})
