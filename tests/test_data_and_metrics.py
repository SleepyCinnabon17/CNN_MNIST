import gzip
import struct

import numpy as np
import pytest

from cnn_mnist.data.augment import random_shift
from cnn_mnist.data.dataloader import DataLoader
from cnn_mnist.data import loader as loader_module
from cnn_mnist.data.loader import (
    MNIST_FILES,
    MNIST_MIRRORS,
    download_mnist,
    one_hot,
    parse_idx_images,
    parse_idx_labels,
    train_val_split,
)
from cnn_mnist.utils.metrics import classification_report, confusion_matrix


def test_idx_parsers_normalize_images_and_read_labels():
    image_payload = bytearray()
    image_payload += struct.pack(">IIII", 2051, 2, 2, 2)
    image_payload += bytes([0, 127, 255, 64, 10, 20, 30, 40])
    label_payload = bytearray()
    label_payload += struct.pack(">II", 2049, 2)
    label_payload += bytes([3, 8])

    images = parse_idx_images(gzip.compress(bytes(image_payload)))
    labels = parse_idx_labels(gzip.compress(bytes(label_payload)))

    assert images.shape == (2, 1, 2, 2)
    assert images.dtype == np.float32
    assert np.isclose(images.max(), 1.0)
    assert labels.tolist() == [3, 8]


def test_one_hot_and_train_val_split_are_deterministic():
    labels = np.array([0, 2, 1])
    assert np.array_equal(one_hot(labels, 3), np.eye(3)[[0, 2, 1]])

    x = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    first = train_val_split(x, y, val_size=3, seed=42)
    second = train_val_split(x, y, val_size=3, seed=42)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)
    assert first[0].shape[0] == 7
    assert first[2].shape[0] == 3


def test_download_mnist_skips_existing_files_and_downloads_missing(tmp_path, monkeypatch):
    existing_keys = {"train_images", "test_labels"}
    calls = []
    for key in existing_keys:
        (tmp_path / MNIST_FILES[key]).write_bytes(b"existing")

    def fake_urlretrieve(url, filename):
        calls.append((url, filename))
        filename.write_bytes(b"downloaded")
        return str(filename), None

    monkeypatch.setattr(loader_module.urllib.request, "urlretrieve", fake_urlretrieve)

    paths = download_mnist(tmp_path)

    assert set(paths) == set(MNIST_FILES)
    assert all(path.exists() for path in paths.values())
    assert {path.name for _, path in calls} == {
        MNIST_FILES[key] for key in MNIST_FILES if key not in existing_keys
    }
    assert all(url.startswith(MNIST_MIRRORS[0]) for url, _ in calls)
    assert all((tmp_path / MNIST_FILES[key]).read_bytes() == b"existing" for key in existing_keys)


def test_dataloader_batches_shuffle_and_optional_train_augmentation():
    x = np.zeros((8, 1, 4, 4), dtype=np.float32)
    y = np.arange(8)
    loader = DataLoader(x, y, batch_size=3, shuffle=True, seed=1)

    batches = list(loader.batches(training=True))
    assert [batch_x.shape[0] for batch_x, _ in batches] == [3, 3, 2]
    assert sorted(np.concatenate([batch_y for _, batch_y in batches]).tolist()) == list(range(8))
    assert not np.array_equal(loader.last_indices, np.arange(8))


def test_random_shift_preserves_shape_and_only_moves_pixels():
    rng = np.random.default_rng(0)
    x = np.zeros((1, 1, 5, 5), dtype=np.float32)
    x[0, 0, 2, 2] = 1.0

    shifted = random_shift(x, max_shift=2, rng=rng)

    assert shifted.shape == x.shape
    assert np.isclose(shifted.sum(), 1.0)


def test_confusion_matrix_and_classification_report():
    y_true = np.array([0, 1, 1, 2], dtype=np.float32)
    y_pred = np.array([0, 1, 2, 2], dtype=np.float32)

    matrix = confusion_matrix(y_true, y_pred, num_classes=3)
    report = classification_report(y_true, y_pred, num_classes=3)

    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert np.isclose(report["precision"][1], 1.0)
    assert np.isclose(report["recall"][1], 0.5)
    assert np.isclose(report["macro_f1"], (1.0 + 2 / 3 + 2 / 3) / 3)


def test_confusion_matrix_rejects_labels_outside_configured_class_range():
    with pytest.raises(ValueError, match="outside"):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), num_classes=3)

    with pytest.raises(ValueError, match="outside"):
        confusion_matrix(np.array([0, 1]), np.array([0, -1]), num_classes=3)
