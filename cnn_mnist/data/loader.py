"""MNIST IDX download and parsing utilities."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Optional, Union

import numpy as np


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _decompress(source: Union[bytes, str, Path]) -> bytes:
    if isinstance(source, bytes):
        return gzip.decompress(source)
    with gzip.open(Path(source), "rb") as handle:
        return handle.read()


def parse_idx_images(source: Union[bytes, str, Path]) -> np.ndarray:
    """Parse IDX image bytes or a ``.gz`` file into ``(N, 1, H, W)`` float32 arrays."""
    raw = _decompress(source)
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"invalid image magic number: {magic}")
    data = np.frombuffer(raw[16:], dtype=np.uint8)
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"expected {expected} image bytes, found {data.size}")
    return data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0


def parse_idx_labels(source: Union[bytes, str, Path]) -> np.ndarray:
    """Parse IDX label bytes or a ``.gz`` file into integer labels."""
    raw = _decompress(source)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"invalid label magic number: {magic}")
    labels = np.frombuffer(raw[8:], dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"expected {count} labels, found {labels.size}")
    return labels.astype(np.int64)


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Return one-hot encoded labels."""
    return np.eye(num_classes, dtype=np.float64)[labels.astype(int)]


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_size: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into deterministic train and validation partitions."""
    if not 0 < val_size < len(x):
        raise ValueError("val_size must be between 1 and len(x)-1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def download_mnist(data_dir: Union[str, Path]) -> dict[str, Path]:
    """Download missing MNIST IDX gzip files to ``data_dir``."""
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {key: directory / filename for key, filename in MNIST_FILES.items()}
    for key, path in paths.items():
        if path.exists():
            continue
        last_error: Optional[Exception] = None
        for mirror in MNIST_MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + MNIST_FILES[key], path)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"could not download {MNIST_FILES[key]}") from last_error
    return paths


def load_mnist(
    data_dir: Union[str, Path] = "./data/MNIST/raw",
    download: bool = True,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST train/test arrays, downloading files when requested."""
    paths = download_mnist(data_dir) if download else {
        key: Path(data_dir) / filename for key, filename in MNIST_FILES.items()
    }
    train_images = parse_idx_images(paths["train_images"])
    train_labels = parse_idx_labels(paths["train_labels"])
    test_images = parse_idx_images(paths["test_images"])
    test_labels = parse_idx_labels(paths["test_labels"])
    if normalize:
        train_images = (train_images - 0.1307) / 0.3081
        test_images = (test_images - 0.1307) / 0.3081
    return train_images, train_labels, test_images, test_labels
