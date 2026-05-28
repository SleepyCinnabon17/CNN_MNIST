"""CNN-MNIST-Scratch: a NumPy-only CNN implementation for MNIST."""

from cnn_mnist.config import TrainConfig
from cnn_mnist.model import Model, build_lenet_model

__all__ = ["Model", "TrainConfig", "build_lenet_model"]
