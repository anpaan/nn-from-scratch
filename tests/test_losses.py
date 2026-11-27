import numpy as np

from engine import MSE, SoftmaxCrossEntropy
from engine.gradcheck import numerical_grad, relative_error

rng = np.random.default_rng(7)


def test_mse_value():
    assert np.isclose(MSE().forward(np.array([1.0, 2.0]), np.array([1.0, 4.0])), 2.0)


def test_mse_grad():
    loss = MSE()
    pred, target = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    loss.forward(pred, target)
    numeric = numerical_grad(lambda: loss.forward(pred, target), pred)
    loss.forward(pred, target)
    assert relative_error(loss.backward(), numeric) < 1e-5


def test_softmax_ce_value_uniform():
    # all-equal logits over 4 classes -> loss = -log(1/4)
    loss = SoftmaxCrossEntropy().forward(np.zeros((3, 4)), np.array([0, 1, 2]))
    assert np.isclose(loss, np.log(4))


def test_softmax_ce_stable_with_big_logits():
    loss = SoftmaxCrossEntropy().forward(np.array([[1000.0, 0.0]]), np.array([0]))
    assert np.isfinite(loss)


def test_softmax_ce_grad():
    loss = SoftmaxCrossEntropy()
    logits, target = rng.normal(size=(5, 3)), np.array([0, 2, 1, 1, 0])
    numeric = numerical_grad(lambda: loss.forward(logits, target), logits)
    loss.forward(logits, target)
    assert relative_error(loss.backward(), numeric) < 1e-5
