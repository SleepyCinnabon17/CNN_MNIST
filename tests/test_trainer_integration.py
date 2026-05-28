import numpy as np

from cnn_mnist.config import TrainConfig
from cnn_mnist.layers import Dense, ReLU
from cnn_mnist.model import Model
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
