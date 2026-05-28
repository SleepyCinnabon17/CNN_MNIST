import time

import numpy as np
import pytest

from cnn_mnist.layers import (
    BatchNorm,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
    ReLU,
    Softmax,
)
from cnn_mnist.loss import SoftmaxCrossEntropyLoss
from cnn_mnist.utils.grad_check import check_layer_gradient, relative_error


def _check_input_gradient(layer, x, dout, h=1e-5):
    layer.forward(x, training=True)
    analytic = layer.backward(dout).copy()
    numeric = np.zeros_like(x)
    iterator = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = x[index]
        x[index] = original + h
        plus = float(np.sum(layer.forward(x, training=True) * dout))
        x[index] = original - h
        minus = float(np.sum(layer.forward(x, training=True) * dout))
        x[index] = original
        numeric[index] = (plus - minus) / (2.0 * h)
        iterator.iternext()
    return relative_error(analytic, numeric)


def test_dense_forward_backward_matches_closed_form():
    layer = Dense(3, 2, rng=np.random.default_rng(0))
    layer.W[...] = np.arange(6, dtype=np.float64).reshape(3, 2) / 10.0
    layer.b[...] = np.array([0.25, -0.5])
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    dout = np.array([[0.2, -0.3], [1.0, 0.5]])

    out = layer.forward(x)
    dx = layer.backward(dout)

    assert np.allclose(out, x @ layer.W + layer.b)
    assert np.allclose(layer.grads()["W"], x.T @ dout)
    assert np.allclose(layer.grads()["b"], dout.sum(axis=0))
    assert np.allclose(dx, dout @ layer.W.T)


def test_relu_flatten_and_softmax_behaviour():
    relu = ReLU()
    x = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu.forward(x), np.array([[0.0, 0.0, 3.0]]))
    assert np.array_equal(relu.backward(np.ones_like(x)), np.array([[0.0, 0.0, 1.0]]))

    flatten = Flatten()
    volume = np.arange(24).reshape(2, 3, 2, 2)
    flat = flatten.forward(volume)
    assert flat.shape == (2, 12)
    assert np.array_equal(flatten.backward(flat), volume)

    probs = Softmax().forward(np.array([[1000.0, 1001.0, 999.0]]))
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()


def test_softmax_backward_matches_finite_difference_jacobian_vector_product():
    rng = np.random.default_rng(2)
    layer = Softmax()
    x = rng.normal(size=(2, 4))
    dout = rng.normal(size=(2, 4))

    error = _check_input_gradient(layer, x, dout)

    assert error < 1e-6


def test_softmax_cross_entropy_known_values_and_gradient():
    loss = SoftmaxCrossEntropyLoss()
    logits = np.zeros((2, 10), dtype=np.float64)
    labels = np.array([0, 3])

    value = loss.forward(logits, labels)
    grad = loss.backward()

    assert np.isclose(value, np.log(10.0))
    assert grad.shape == logits.shape
    assert np.allclose(grad.sum(axis=1), 0.0)
    assert np.isclose(grad[0, 0], -0.45)
    assert np.isclose(grad[0, 1], 0.05)


def test_softmax_cross_entropy_gradient_matches_loss_when_eps_is_nonzero():
    loss = SoftmaxCrossEntropyLoss(eps=0.1)
    logits = np.array([[1.2, -0.4, 0.3]], dtype=np.float64)
    labels = np.array([2])
    loss.forward(logits, labels)
    analytic = loss.backward().copy()
    numeric = np.zeros_like(logits)
    h = 1e-5

    for index in np.ndindex(logits.shape):
        original = logits[index]
        logits[index] = original + h
        plus = loss.forward(logits, labels)
        logits[index] = original - h
        minus = loss.forward(logits, labels)
        logits[index] = original
        numeric[index] = (plus - minus) / (2.0 * h)

    assert relative_error(analytic, numeric) < 1e-6


def test_dropout_is_training_only_and_scaled():
    rng = np.random.default_rng(123)
    dropout = Dropout(rate=0.25, rng=rng)
    x = np.ones((2000,), dtype=np.float64)

    train_out = dropout.forward(x, training=True)
    eval_out = dropout.forward(x, training=False)

    assert np.isclose(train_out.mean(), 1.0, atol=0.08)
    assert 0.20 < np.mean(train_out == 0.0) < 0.30
    assert np.array_equal(eval_out, x)


def test_batchnorm_normalizes_training_batch_and_uses_running_stats():
    bn = BatchNorm(num_features=3, rng=np.random.default_rng(0), momentum=0.5)
    x = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 7.0], [5.0, 6.0, 11.0]])

    out = bn.forward(x, training=True)
    eval_out = bn.forward(x, training=False)

    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.var(axis=0), 1.0, atol=1e-4)
    assert eval_out.shape == x.shape


def test_batchnorm_rejects_unsupported_inputs_and_backward_before_forward():
    bn = BatchNorm(num_features=3)

    with pytest.raises(ValueError, match="2D or 4D"):
        bn.forward(np.zeros((1, 3, 2)))
    with pytest.raises(RuntimeError, match="before training forward"):
        bn.backward(np.zeros((1, 3)))

    bn._shape = (1, 3)
    bn._x_hat = np.zeros((1, 3))
    with pytest.raises(RuntimeError, match="cache is incomplete"):
        bn.backward(np.zeros((1, 3)))


def test_batchnorm_backward_matches_parameter_and_input_gradients():
    rng = np.random.default_rng(8)
    bn = BatchNorm(num_features=3, rng=rng)
    x = rng.normal(loc=4.0, scale=1.5, size=(4, 3))
    dout = rng.normal(size=(4, 3))

    param_errors = check_layer_gradient(bn, x, dout)
    input_error = _check_input_gradient(bn, x, dout)

    assert param_errors["gamma"] < 1e-5
    assert param_errors["beta"] < 1e-8
    assert input_error < 1e-5


def test_batchnorm_backward_supports_4d_conv_activations():
    rng = np.random.default_rng(9)
    bn = BatchNorm(num_features=3, rng=rng)
    x = rng.normal(size=(2, 3, 2, 2))
    dout = rng.normal(size=(2, 3, 2, 2))

    error = _check_input_gradient(bn, x, dout)

    assert error < 1e-5


def test_conv_batchnorm_relu_sequence_supports_4d_forward_backward():
    rng = np.random.default_rng(12)
    conv = Conv2D(1, 2, 3, padding="same", rng=rng)
    bn = BatchNorm(num_features=2, rng=rng)
    relu = ReLU()
    x = rng.normal(loc=3.0, scale=2.0, size=(2, 1, 4, 4))

    out = relu.forward(bn.forward(conv.forward(x, training=True), training=True), training=True)
    grad = conv.backward(bn.backward(relu.backward(np.ones_like(out))))

    assert out.shape == (2, 2, 4, 4)
    assert grad.shape == x.shape
    assert np.isfinite(grad).all()


def test_maxpool_forward_and_backward_routes_argmax_only():
    pool = MaxPool2D(pool_size=2, stride=2)
    x = np.array([[[[1.0, 5.0], [2.0, 3.0]]]])

    out = pool.forward(x)
    dx = pool.backward(np.ones_like(out))

    assert np.array_equal(out, np.array([[[[5.0]]]]))
    assert np.array_equal(dx, np.array([[[[0.0, 1.0], [0.0, 0.0]]]]))


def test_maxpool_backward_vectorized_performance_budget():
    pool = MaxPool2D(pool_size=2, stride=2)
    x = np.random.default_rng(0).normal(size=(16, 64, 28, 28))
    out = pool.forward(x)

    start = time.perf_counter()
    dx = pool.backward(np.ones_like(out))
    elapsed = time.perf_counter() - start

    assert dx.shape == x.shape
    assert elapsed < 0.08


def test_maxpool_forward_caches_contiguous_input_for_strided_windows():
    pool = MaxPool2D(pool_size=2, stride=2)
    x = np.arange(32, dtype=np.float64).reshape(1, 1, 4, 8)[:, :, :, ::2]

    out = pool.forward(x)

    assert not x.flags.c_contiguous
    assert pool._x is not None
    assert pool._x.flags.c_contiguous
    assert out.shape == (1, 1, 2, 2)


def test_conv2d_forward_shape_and_small_gradient_check():
    rng = np.random.default_rng(7)
    conv = Conv2D(
        in_channels=1,
        num_filters=2,
        kernel_size=3,
        stride=1,
        padding="same",
        rng=rng,
    )
    x = rng.normal(size=(2, 1, 4, 4))

    out = conv.forward(x)
    errors = check_layer_gradient(conv, x, rng.normal(size=out.shape), h=1e-5)

    assert out.shape == (2, 2, 4, 4)
    assert errors["W"] < 1e-4
    assert errors["b"] < 1e-8


def test_conv2d_rejects_bad_padding_and_backward_before_forward():
    with pytest.raises(ValueError, match="padding"):
        Conv2D(1, 2, 3, padding="mirror")

    conv = Conv2D(1, 2, 3)
    with pytest.raises(RuntimeError, match="before forward"):
        conv.backward(np.zeros((1, 2, 2, 2)))


def test_conv2d_cached_im2col_matrix_is_read_only():
    rng = np.random.default_rng(13)
    conv = Conv2D(1, 2, 3, padding="same", rng=rng)

    conv.forward(rng.normal(size=(1, 1, 4, 4)))

    assert conv._cols is not None
    assert not conv._cols.flags.writeable


def test_conv2d_valid_padding_input_gradient_matches_finite_difference():
    rng = np.random.default_rng(10)
    conv = Conv2D(1, 2, 3, stride=1, padding="valid", rng=rng)
    x = rng.normal(size=(1, 1, 4, 4))
    dout = rng.normal(size=conv.forward(x).shape)

    error = _check_input_gradient(conv, x, dout)

    assert error < 1e-6


def test_conv2d_asymmetric_same_padding_input_gradient_matches_finite_difference():
    rng = np.random.default_rng(11)
    conv = Conv2D(1, 2, 3, stride=2, padding="same", rng=rng)
    x = rng.normal(size=(1, 1, 4, 4))
    dout = rng.normal(size=conv.forward(x).shape)

    error = _check_input_gradient(conv, x, dout)

    assert conv._pad[:4] == (0, 1, 0, 1)
    assert error < 1e-6
