import numpy as np

from cnn_mnist.inference import predict
from cnn_mnist.utils.visualization import (
    plot_confusion_matrix,
    plot_conv_filters,
    plot_history,
    plot_sample_predictions,
)


class FixedModel:
    def predict_proba(self, x):
        assert x.shape == (1, 1, 28, 28)
        return np.array([[0.1, 0.7, 0.2]])


def test_predict_normalizes_raw_image_and_returns_label_and_probabilities():
    image = np.full((28, 28), 255, dtype=np.uint8)

    label, probs = predict(image, model=FixedModel())

    assert label == 1
    assert np.allclose(probs.sum(), 1.0)


def test_visualization_artifacts_are_saved(tmp_path):
    history = {
        "train_loss": [2.0, 1.0],
        "val_loss": [2.2, 1.2],
        "train_accuracy": [0.2, 0.8],
        "val_accuracy": [0.1, 0.7],
    }
    images = np.zeros((16, 1, 28, 28), dtype=np.float32)
    filters = np.zeros((4, 1, 3, 3), dtype=np.float32)

    plot_history(history, tmp_path)
    plot_confusion_matrix(np.eye(3, dtype=int), tmp_path)
    plot_sample_predictions(images, np.arange(16) % 10, np.arange(16) % 10, tmp_path)
    plot_conv_filters(filters, tmp_path)

    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "accuracy_curve.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "sample_predictions.png").exists()
    assert (tmp_path / "conv1_filters.png").exists()
