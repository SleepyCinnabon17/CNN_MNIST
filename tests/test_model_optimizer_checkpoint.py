import numpy as np
import pytest

from cnn_mnist.config import TrainConfig
from cnn_mnist.layers import Conv2D, Dense, ReLU
from cnn_mnist.model import Model, build_lenet_model
from cnn_mnist.optimizers import Adam, SGDMomentum, cosine_lr, step_decay_lr
from cnn_mnist.utils.checkpoint import load_model, save_model


def test_model_forward_backward_predict_and_state_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    model = Model([Dense(4, 5, rng=rng), ReLU(), Dense(5, 3, rng=rng)])
    x = rng.normal(size=(2, 4))
    dout = rng.normal(size=(2, 3))

    logits = model.forward(x, training=True)
    dx = model.backward(dout)
    path = tmp_path / "weights.npz"
    save_model(model, path)

    clone = Model([Dense(4, 5, rng=rng), ReLU(), Dense(5, 3, rng=rng)])
    load_model(clone, path)

    assert logits.shape == (2, 3)
    assert dx.shape == x.shape
    assert model.predict_proba(x).shape == (2, 3)
    assert model.predict(x).shape == (2,)
    for key, value in model.state_dict().items():
        assert np.array_equal(value, clone.state_dict()[key])


def test_sgd_momentum_updates_parameters_and_weight_decay():
    w = np.array([1.0, -2.0])
    params = {"W": w}
    grads = {"W": np.array([0.5, -0.25])}
    opt = SGDMomentum(lr=0.1, momentum=0.9, weight_decay=0.1)

    opt.step(params, grads)

    expected_grad = grads["W"] + 0.1 * np.array([1.0, -2.0])
    assert np.allclose(w, np.array([1.0, -2.0]) - 0.1 * expected_grad)


def test_sgd_momentum_accumulates_velocity_across_steps():
    w = np.array([1.0, -2.0])
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    grads = [np.array([0.5, -0.25]), np.array([0.1, 0.2])]
    expected = w.copy()
    velocity = np.zeros_like(w)

    for grad in grads:
        opt.step({"W": w}, {"W": grad})
        velocity = opt.momentum * velocity - opt.lr * grad
        expected += velocity

    assert np.allclose(w, expected)
    assert np.allclose(opt.velocity["W"], velocity)


def test_adam_optimizer_moves_against_gradient():
    w = np.array([1.0, -1.0])
    opt = Adam(lr=0.01)

    opt.step({"W": w}, {"W": np.array([0.25, -0.25])})

    assert w[0] < 1.0
    assert w[1] > -1.0


def test_adam_optimizer_matches_two_step_bias_corrected_reference():
    w = np.array([1.0, -1.0])
    opt = Adam(lr=0.01, beta1=0.8, beta2=0.9, eps=1e-8)
    grads = [np.array([0.25, -0.5]), np.array([0.1, 0.2])]
    expected = w.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    for step, grad in enumerate(grads, start=1):
        opt.step({"W": w}, {"W": grad})
        m = opt.beta1 * m + (1.0 - opt.beta1) * grad
        v = opt.beta2 * v + (1.0 - opt.beta2) * (grad * grad)
        m_hat = m / (1.0 - opt.beta1**step)
        v_hat = v / (1.0 - opt.beta2**step)
        expected -= opt.lr * m_hat / (np.sqrt(v_hat) + opt.eps)

    assert np.allclose(w, expected)
    assert opt.t == 2


def test_learning_rate_schedules():
    assert step_decay_lr(0.1, epoch=0, decay_factor=0.5, decay_epochs=5) == 0.1
    assert step_decay_lr(0.1, epoch=4, decay_factor=0.5, decay_epochs=5) == 0.1
    assert step_decay_lr(0.1, epoch=5, decay_factor=0.5, decay_epochs=5) == 0.05
    assert np.isclose(cosine_lr(0.1, epoch=10, total_epochs=10), 0.0)


def test_model_builder_supports_compact_default_and_srs_reference():
    compact = build_lenet_model(TrainConfig(architecture="compact_lenet"))
    srs = build_lenet_model(TrainConfig(architecture="srs_reference"))
    x = np.zeros((2, 1, 28, 28), dtype=np.float64)

    compact_convs = [layer for layer in compact.layers if isinstance(layer, Conv2D)]
    srs_convs = [layer for layer in srs.layers if isinstance(layer, Conv2D)]

    assert len(compact_convs) == 2
    assert len(srs_convs) == 4
    assert compact.forward(x, training=False).shape == (2, 10)
    assert srs.forward(x, training=False).shape == (2, 10)


def test_train_config_num_classes_controls_model_output_width():
    model = build_lenet_model(TrainConfig(architecture="compact_lenet", num_classes=3))
    x = np.zeros((2, 1, 28, 28), dtype=np.float64)

    assert model.forward(x, training=False).shape == (2, 3)


def test_load_state_dict_rejects_missing_extra_and_shape_mismatches():
    model = Model([Dense(2, 3, rng=np.random.default_rng(0))])
    state = model.state_dict()

    missing = {key: value for key, value in state.items() if key != "0.b"}
    with pytest.raises(ValueError, match="missing"):
        model.load_state_dict(missing)

    extra = dict(state)
    extra["999.W"] = np.zeros((1,))
    with pytest.raises(ValueError, match="extra"):
        model.load_state_dict(extra)

    wrong_shape = dict(state)
    wrong_shape["0.W"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict(wrong_shape)
