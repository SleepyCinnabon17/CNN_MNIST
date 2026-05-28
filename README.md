# CNN-MNIST-Scratch

NumPy-only convolutional neural network for MNIST, implemented without TensorFlow,
PyTorch, Keras, JAX, automatic differentiation, or prebuilt convolution/pooling
operators.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest -m "not slow"
```

Full MNIST acceptance checks are opt-in because they run 20-epoch training jobs:

```bash
python -m pytest tests/test_acceptance_slow.py -m slow --run-full-mnist
python -m pytest tests/test_acceptance_slow.py::test_full_mnist_reproducibility_acceptance --run-full-mnist-repro
```

## Train

```bash
python -m cnn_mnist.train --epochs 20 --batch-size 64
```

Training downloads MNIST IDX files into `./data/MNIST/raw`, scales image bytes
from `[0, 255]` to `[0.0, 1.0]`, trains the default CPU-friendly compact LeNet
CNN, saves the best checkpoint to `./checkpoints/best_model.npz`, and writes
artifacts to `./outputs`.

The larger SRS reference architecture remains available:

```bash
python -m cnn_mnist.train --architecture srs_reference --epochs 20 --batch-size 64
```

Use `--normalize` only when you want the additional MNIST mean/std
standardization step after the default `[0.0, 1.0]` scaling.

Expected output artifacts:

- `training_history.json`
- `test_metrics.json`
- `loss_curve.png`
- `accuracy_curve.png`
- `confusion_matrix.png`
- `sample_predictions.png`
- `conv1_filters.png`

## Inference

```bash
python -m cnn_mnist.inference path/to/image.npy --weights ./checkpoints/best_model.npz
```

The image file must contain one raw `28x28` NumPy array. Pixel values may be
either `[0, 255]` or normalized `[0.0, 1.0]`. Programmatic calls to
`predict()` require either a loaded model or a weights path.
