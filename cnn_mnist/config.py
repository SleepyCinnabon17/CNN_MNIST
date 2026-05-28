"""Central training configuration for CNN-MNIST-Scratch."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters and artifact locations used by training and evaluation."""

    data_dir: str = "./data/MNIST/raw"
    val_size: int = 10_000
    seed: int = 42
    num_classes: int = 10
    normalize: bool = False
    architecture: str = "compact_lenet"
    optimizer: str = "sgd_momentum"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "step"
    lr_decay_factor: float = 0.5
    lr_decay_epochs: int = 5
    batch_size: int = 64
    epochs: int = 20
    dropout_conv: float = 0.25
    dropout_fc: float = 0.50
    augment: bool = True
    max_shift: int = 2
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    use_tqdm: bool = True
