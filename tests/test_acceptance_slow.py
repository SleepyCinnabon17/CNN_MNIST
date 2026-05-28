import os
import time
from pathlib import Path

import numpy as np
import pytest

from cnn_mnist.config import TrainConfig
from cnn_mnist.data.loader import load_mnist, train_val_split
from cnn_mnist.inference import predict
from cnn_mnist.model import build_lenet_model
from cnn_mnist.trainer import Trainer
from cnn_mnist.utils.checkpoint import load_model


@pytest.mark.slow
def test_full_mnist_training_acceptance(tmp_path):
    if os.environ.get("RUN_FULL_MNIST") != "1":
        pytest.skip("Set RUN_FULL_MNIST=1 to run the full 20-epoch MNIST acceptance test.")

    config = TrainConfig(
        epochs=20,
        batch_size=64,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "outputs"),
        use_tqdm=False,
    )
    train_images, train_labels, test_images, test_labels = load_mnist(config.data_dir)
    train_x, train_y, val_x, val_y = train_val_split(
        train_images,
        train_labels,
        val_size=config.val_size,
        seed=config.seed,
    )
    model = build_lenet_model(config)
    trainer = Trainer(model, config)

    trainer.fit(train_x, train_y, val_x, val_y)
    load_model(model, Path(config.checkpoint_dir) / "best_model.npz")
    metrics = trainer.evaluate(test_images, test_labels)

    assert metrics["accuracy"] >= 0.98
    assert (Path(config.output_dir) / "training_history.json").exists()
    assert (Path(config.output_dir) / "loss_curve.png").exists()
    assert (Path(config.output_dir) / "accuracy_curve.png").exists()
    assert (Path(config.checkpoint_dir) / "best_model.npz").exists()


@pytest.mark.slow
def test_full_mnist_reproducibility_acceptance(tmp_path):
    if os.environ.get("RUN_FULL_MNIST_REPRO") != "1":
        pytest.skip("Set RUN_FULL_MNIST_REPRO=1 to run two full reproducibility runs.")

    accuracies = []
    for run in range(2):
        config = TrainConfig(
            epochs=20,
            batch_size=64,
            checkpoint_dir=str(tmp_path / f"checkpoints_{run}"),
            output_dir=str(tmp_path / f"outputs_{run}"),
            use_tqdm=False,
        )
        train_images, train_labels, test_images, test_labels = load_mnist(config.data_dir)
        train_x, train_y, val_x, val_y = train_val_split(
            train_images,
            train_labels,
            val_size=config.val_size,
            seed=config.seed,
        )
        model = build_lenet_model(config)
        trainer = Trainer(model, config)
        trainer.fit(train_x, train_y, val_x, val_y)
        load_model(model, Path(config.checkpoint_dir) / "best_model.npz")
        accuracies.append(trainer.evaluate(test_images, test_labels)["accuracy"])

    assert abs(accuracies[0] - accuracies[1]) <= 0.001


@pytest.mark.slow
def test_single_image_inference_latency_acceptance():
    if os.environ.get("RUN_FULL_MNIST") != "1":
        pytest.skip("Set RUN_FULL_MNIST=1 after training artifacts exist to test latency.")

    image = np.zeros((28, 28), dtype=np.float32)
    weights = Path("./checkpoints/best_model.npz")
    if not weights.exists():
        pytest.skip("Train first with `python -m cnn_mnist.train --epochs 20`.")
    model = build_lenet_model(TrainConfig())
    load_model(model, weights)

    start = time.perf_counter()
    label, probs = predict(image, model=model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert 0 <= label <= 9
    assert np.isclose(probs.sum(), 1.0)
    assert elapsed_ms <= 100.0
