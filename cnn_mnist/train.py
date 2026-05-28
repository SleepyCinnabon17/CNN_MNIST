"""Command-line entry point for MNIST training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnn_mnist.config import TrainConfig
from cnn_mnist.data.loader import load_mnist, train_val_split
from cnn_mnist.model import build_lenet_model
from cnn_mnist.trainer import Trainer
from cnn_mnist.utils.checkpoint import load_model
from cnn_mnist.utils.visualization import (
    plot_confusion_matrix,
    plot_conv_filters,
    plot_sample_predictions,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a NumPy CNN on MNIST.")
    parser.add_argument("--data-dir", default="./data/MNIST/raw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd_momentum", "adam"], default="sgd_momentum")
    parser.add_argument(
        "--architecture",
        choices=["compact_lenet", "srs_reference"],
        default="compact_lenet",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="apply MNIST mean/std standardization after automatic [0, 1] pixel scaling",
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and save final artifacts."""
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        architecture=args.architecture,
        normalize=args.normalize,
        augment=not args.no_augment,
        use_tqdm=not args.no_tqdm,
    )
    train_images, train_labels, test_images, test_labels = load_mnist(
        config.data_dir,
        download=True,
        normalize=config.normalize,
    )
    train_x, train_y, val_x, val_y = train_val_split(
        train_images,
        train_labels,
        val_size=config.val_size,
        seed=config.seed,
    )
    model = build_lenet_model(config)
    trainer = Trainer(model, config)
    trainer.fit(train_x, train_y, val_x, val_y)
    best_path = Path(config.checkpoint_dir) / "best_model.npz"
    load_model(model, best_path)
    test_metrics = trainer.evaluate(test_images, test_labels)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(test_metrics["confusion_matrix"], output_dir)
    plot_sample_predictions(test_images[:16], test_labels[:16], test_metrics["predictions"][:16], output_dir)
    first_conv = next(layer for layer in model.layers if hasattr(layer, "W") and layer.W.ndim == 4)
    plot_conv_filters(first_conv.W, output_dir)
    serializable = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "precision": test_metrics["precision"].tolist(),
        "recall": test_metrics["recall"].tolist(),
        "f1": test_metrics["f1"].tolist(),
        "macro_f1": float(test_metrics["macro_f1"]),
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
