import numpy as np

from cnn_mnist.config import TrainConfig
from cnn_mnist.layers import Dense, ReLU
from cnn_mnist.model import Model
from cnn_mnist import trainer as trainer_module
from cnn_mnist.trainer import Trainer


def test_trainer_fit_on_small_synthetic_problem_reduces_loss(tmp_path):
    rng = np.random.default_rng(4)
    x0 = rng.normal(loc=-1.0, scale=0.2, size=(40, 2))
    x1 = rng.normal(loc=1.0, scale=0.2, size=(40, 2))
    x = np.vstack([x0, x1]).astype(np.float64)
    y = np.array([0] * 40 + [1] * 40)

    model = Model([Dense(2, 8, rng=rng), ReLU(), Dense(8, 2, rng=rng)])
    config = TrainConfig(
        epochs=5,
        batch_size=16,
        lr=0.1,
        output_dir=str(tmp_path / "outputs"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_tqdm=False,
    )
    trainer = Trainer(model, config)

    history = trainer.fit(x, y, x, y)
    metrics = trainer.evaluate(x, y)

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert metrics["accuracy"] > 0.9
    assert (tmp_path / "checkpoints" / "best_model.npz").exists()
    assert (tmp_path / "outputs" / "training_history.json").exists()


def test_trainer_evaluate_uses_configured_num_classes_for_metrics():
    class TwoClassLogitModel:
        def forward(self, x, training=True):
            del training
            logits = np.zeros((len(x), 2), dtype=np.float64)
            logits[:, 0] = 1.0
            return logits

    trainer = Trainer(TwoClassLogitModel(), TrainConfig(batch_size=4))
    metrics = trainer.evaluate(np.zeros((3, 2)), np.zeros(3, dtype=int))

    assert metrics["confusion_matrix"].shape == (10, 10)
    assert metrics["precision"].shape == (10,)


def test_direct_train_epoch_calls_advance_deterministic_shuffle_seed(monkeypatch):
    class CapturingDataLoader:
        seeds = []

        def __init__(self, x, y, batch_size, shuffle, seed, augment, max_shift):
            del batch_size, shuffle, augment, max_shift
            self.x = x
            self.y = y
            self.seeds.append(seed)

        def batches(self, training=True):
            del training
            yield self.x, self.y

    monkeypatch.setattr(trainer_module, "DataLoader", CapturingDataLoader)
    rng = np.random.default_rng(5)
    model = Model([Dense(2, 2, rng=rng)])
    trainer = Trainer(model, TrainConfig(batch_size=4, lr=0.01, use_tqdm=False))
    x = rng.normal(size=(4, 2))
    y = np.array([0, 1, 0, 1])

    trainer.train_epoch(x, y)
    trainer.train_epoch(x, y)

    assert CapturingDataLoader.seeds == [42, 43]
