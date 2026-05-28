"""Training and evaluation loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from cnn_mnist.config import TrainConfig
from cnn_mnist.data.dataloader import DataLoader
from cnn_mnist.loss import SoftmaxCrossEntropyLoss
from cnn_mnist.model import Model
from cnn_mnist.optimizers import Adam, SGDMomentum, cosine_lr, step_decay_lr
from cnn_mnist.utils.checkpoint import save_model
from cnn_mnist.utils.metrics import accuracy, classification_report
from cnn_mnist.utils.visualization import plot_history


class Trainer:
    """Coordinate mini-batch training, validation, checkpointing, and history."""

    def __init__(self, model: Model, config: TrainConfig) -> None:
        self.model = model
        self.config = config
        self.loss = SoftmaxCrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self._epoch_index = 0

    def _build_optimizer(self) -> Union[SGDMomentum, Adam]:
        if self.config.optimizer == "sgd_momentum":
            return SGDMomentum(self.config.lr, self.config.momentum, self.config.weight_decay)
        if self.config.optimizer == "adam":
            return Adam(lr=self.config.lr, weight_decay=self.config.weight_decay)
        raise ValueError(f"unknown optimizer: {self.config.optimizer}")

    def _set_epoch_lr(self, epoch: int) -> None:
        if self.config.lr_schedule == "step":
            lr = step_decay_lr(
                self.config.lr,
                epoch,
                self.config.lr_decay_factor,
                self.config.lr_decay_epochs,
            )
        elif self.config.lr_schedule == "cosine":
            lr = cosine_lr(self.config.lr, epoch, self.config.epochs)
        else:
            lr = self.config.lr
        self.optimizer.lr = lr

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> dict[str, list[float]]:
        """Train for configured epochs and save best validation checkpoint."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        best_val_accuracy = -np.inf
        for epoch in range(self.config.epochs):
            self._set_epoch_lr(epoch)
            train_metrics = self.train_epoch(train_x, train_y)
            val_metrics = self.evaluate(val_x, val_y)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                save_model(self.model, Path(self.config.checkpoint_dir) / "best_model.npz")
            print(
                f"epoch={epoch + 1}/{self.config.epochs} "
                f"loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )
        self._save_history()
        plot_history(self.history, self.config.output_dir)
        return self.history

    def train_epoch(self, train_x: np.ndarray, train_y: np.ndarray) -> dict[str, float]:
        """Run one training epoch."""
        loader = DataLoader(
            train_x,
            train_y,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed + self._epoch_index,
            augment=self.config.augment and train_x.ndim == 4,
            max_shift=self.config.max_shift,
        )
        self._epoch_index += 1
        total_loss = 0.0
        total_seen = 0
        correct = 0
        iterator = loader.batches(training=True)
        if self.config.use_tqdm:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=int(np.ceil(len(train_x) / self.config.batch_size)))
            except Exception:
                pass
        for batch_x, batch_y in iterator:
            logits = self.model.forward(batch_x, training=True)
            batch_loss = self.loss.forward(logits, batch_y)
            dout = self.loss.backward()
            self.model.backward(dout)
            self.optimizer.step(self.model.params(), self.model.grads())
            total_loss += batch_loss * len(batch_x)
            total_seen += len(batch_x)
            correct += int(np.sum(np.argmax(logits, axis=1) == batch_y.astype(int)))
        return {"loss": total_loss / total_seen, "accuracy": correct / total_seen}

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Evaluate loss, accuracy, and per-class metrics in evaluation mode."""
        loader = DataLoader(x, y, batch_size=self.config.batch_size, shuffle=False, seed=self.config.seed)
        total_loss = 0.0
        total_seen = 0
        preds: list[np.ndarray] = []
        for batch_x, batch_y in loader.batches(training=False):
            logits = self.model.forward(batch_x, training=False)
            batch_loss = self.loss.forward(logits, batch_y)
            total_loss += batch_loss * len(batch_x)
            total_seen += len(batch_x)
            preds.append(np.argmax(logits, axis=1))
        y_pred = np.concatenate(preds)
        report = classification_report(y, y_pred, num_classes=self.config.num_classes)
        return {
            "loss": total_loss / total_seen,
            "accuracy": accuracy(y, y_pred),
            "predictions": y_pred,
            **report,
        }

    def _save_history(self) -> None:
        path = Path(self.config.output_dir) / "training_history.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
