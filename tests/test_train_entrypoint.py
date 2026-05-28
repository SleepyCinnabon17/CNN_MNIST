import sys
from types import SimpleNamespace

import numpy as np

import cnn_mnist.train as train


def test_train_main_does_not_touch_global_numpy_seed(monkeypatch, tmp_path):
    def fail_if_global_seed_is_used(*args, **kwargs):
        raise AssertionError("train.main must not call np.random.seed")

    class FakeTrainer:
        def __init__(self, model, config):
            self.model = model
            self.config = config

        def fit(self, train_x, train_y, val_x, val_y):
            return {"train_loss": [1.0], "val_loss": [1.0]}

        def evaluate(self, x, y):
            return {
                "loss": 0.5,
                "accuracy": 1.0,
                "precision": np.ones(2),
                "recall": np.ones(2),
                "f1": np.ones(2),
                "macro_f1": 1.0,
                "confusion_matrix": np.eye(2, dtype=int),
                "predictions": np.array([0, 1]),
            }

    model = SimpleNamespace(
        layers=[SimpleNamespace(W=np.zeros((1, 1, 3, 3)))],
    )
    config = train.TrainConfig(
        epochs=1,
        batch_size=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "outputs"),
        use_tqdm=False,
    )

    monkeypatch.setattr(np.random, "seed", fail_if_global_seed_is_used)
    monkeypatch.setattr(train, "TrainConfig", lambda **kwargs: config)
    monkeypatch.setattr(
        train,
        "load_mnist",
        lambda *args, **kwargs: (
            np.zeros((2, 1, 28, 28)),
            np.array([0, 1]),
            np.zeros((2, 1, 28, 28)),
            np.array([0, 1]),
        ),
    )
    monkeypatch.setattr(
        train,
        "train_val_split",
        lambda x, y, val_size, seed: (x, y, x, y),
    )
    monkeypatch.setattr(train, "build_lenet_model", lambda cfg: model)
    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "plot_confusion_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "plot_sample_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "plot_conv_filters", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["train.py", "--epochs", "1", "--no-tqdm"])

    train.main()

    assert (tmp_path / "outputs" / "test_metrics.json").exists()
