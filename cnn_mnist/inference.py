"""Single-image inference helpers and CLI."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from cnn_mnist.config import TrainConfig
from cnn_mnist.model import Model, build_lenet_model
from cnn_mnist.utils.checkpoint import load_model


def predict(
    image: np.ndarray,
    model: Optional[Model] = None,
    weights_path: Optional[str] = None,
) -> tuple[int, np.ndarray]:
    """Predict one raw ``28x28`` image and return ``(label, probabilities)``."""
    if image.shape != (28, 28):
        raise ValueError("predict expects a raw 28x28 image")
    if model is None:
        model = build_lenet_model(TrainConfig())
    if weights_path is not None:
        load_model(model, weights_path)
    x: np.ndarray = image.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    probs = model.predict_proba(x.reshape(1, 1, 28, 28))[0]
    return int(np.argmax(probs)), probs


def main() -> None:
    """Run inference for a ``.npy`` file containing one 28x28 image."""
    parser = argparse.ArgumentParser(description="Predict one MNIST digit from a .npy image.")
    parser.add_argument("image_npy")
    parser.add_argument("--weights", default="./checkpoints/best_model.npz")
    args = parser.parse_args()
    label, probs = predict(np.load(args.image_npy), weights_path=args.weights)
    print({"label": label, "probabilities": probs.tolist()})


if __name__ == "__main__":
    main()
