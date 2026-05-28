import numpy as np

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


def test_adam_optimizer_moves_against_gradient():
    w = np.array([1.0, -1.0])
    opt = Adam(lr=0.01)

    opt.step({"W": w}, {"W": np.array([0.25, -0.25])})

    assert w[0] < 1.0
    assert w[1] > -1.0


def test_learning_rate_schedules():
    assert step_decay_lr(0.1, epoch=0, decay_factor=0.5, decay_epochs=5) == 0.1
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
