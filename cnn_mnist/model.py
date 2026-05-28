"""Sequential model orchestration."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from cnn_mnist.config import TrainConfig
from cnn_mnist.layers import Conv2D, Dense, Dropout, Flatten, LayerBase, MaxPool2D, ReLU


class Model:
    """Sequential container for layers with prediction and state helpers."""

    def __init__(self, layers: Iterable[LayerBase]) -> None:
        self.layers: List[LayerBase] = list(layers)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Run a forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Run a backward pass through all layers."""
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> dict[str, np.ndarray]:
        """Return a flat dict of mutable parameter arrays."""
        flat: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers):
            for name, value in layer.params().items():
                flat[f"{index}.{name}"] = value
        return flat

    def grads(self) -> dict[str, np.ndarray]:
        """Return a flat dict of parameter gradients."""
        flat: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers):
            for name, value in layer.grads().items():
                flat[f"{index}.{name}"] = value
        return flat

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a copy-safe mapping of all trainable parameters."""
        return {key: value.copy() for key, value in self.params().items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Load parameters from a mapping keyed like ``state_dict``."""
        params = self.params()
        missing = set(params) - set(state)
        extra = set(state) - set(params)
        if missing or extra:
            raise ValueError(f"state mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        for key, target in params.items():
            if target.shape != state[key].shape:
                raise ValueError(f"shape mismatch for {key}: {target.shape} vs {state[key].shape}")
            target[...] = state[key]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities in evaluation mode."""
        logits = self.forward(x, training=False)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        return np.argmax(self.predict_proba(x), axis=1)


def build_lenet_model(config: Optional[TrainConfig] = None) -> Model:
    """Build a LeNet-style CNN architecture."""
    config = config or TrainConfig()
    rng = np.random.default_rng(config.seed)
    if config.architecture == "compact_lenet":
        return Model(
            [
                Conv2D(1, 16, 3, stride=1, padding="same", rng=rng),
                ReLU(),
                MaxPool2D(2, stride=2),
                Dropout(0.10, rng=rng),
                Conv2D(16, 32, 3, stride=1, padding="same", rng=rng),
                ReLU(),
                MaxPool2D(2, stride=2),
                Flatten(),
                Dense(32 * 7 * 7, 128, rng=rng),
                ReLU(),
                Dropout(0.25, rng=rng),
                Dense(128, 10, rng=rng),
            ]
        )
    if config.architecture != "srs_reference":
        raise ValueError("architecture must be 'compact_lenet' or 'srs_reference'")
    return Model(
        [
            Conv2D(1, 32, 3, stride=1, padding="same", rng=rng),
            ReLU(),
            Conv2D(32, 64, 3, stride=1, padding="same", rng=rng),
            ReLU(),
            MaxPool2D(2, stride=2),
            Dropout(config.dropout_conv, rng=rng),
            Conv2D(64, 128, 3, stride=1, padding="same", rng=rng),
            ReLU(),
            Conv2D(128, 128, 3, stride=1, padding="same", rng=rng),
            ReLU(),
            MaxPool2D(2, stride=2),
            Dropout(config.dropout_conv, rng=rng),
            Flatten(),
            Dense(128 * 7 * 7, 1024, rng=rng),
            ReLU(),
            Dropout(config.dropout_fc, rng=rng),
            Dense(1024, 10, rng=rng),
        ]
    )
