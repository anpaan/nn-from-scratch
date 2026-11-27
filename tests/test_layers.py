import numpy as np

from engine import Linear, ReLU, Sigmoid, Tanh
from engine.gradcheck import numerical_grad, relative_error

rng = np.random.default_rng(42)


def check_input_grad(layer, x):
    """Use loss = sum(out * R) with random R, so dLoss/dOut = R."""
    R = rng.normal(size=layer.forward(x).shape)
    analytic = layer.backward(R)
    numeric = numerical_grad(lambda: float(np.sum(layer.forward(x) * R)), x)
    assert relative_error(analytic, numeric) < 1e-5


def test_linear_forward_shape():
    layer = Linear(3, 5, seed=0)
    assert layer.forward(rng.normal(size=(4, 3))).shape == (4, 5)


def test_linear_input_grad():
    check_input_grad(Linear(3, 4, seed=0), rng.normal(size=(5, 3)))


def test_linear_param_grads():
    layer = Linear(3, 4, seed=0)
    x = rng.normal(size=(5, 3))
    R = rng.normal(size=(5, 4))
    layer.forward(x)
    layer.backward(R)
    f = lambda: float(np.sum(layer.forward(x) * R))
    assert relative_error(layer.dW, numerical_grad(f, layer.W)) < 1e-5
    assert relative_error(layer.db, numerical_grad(f, layer.b)) < 1e-5


def test_relu_grad():
    # keep values away from 0, where ReLU has a kink
    x = rng.uniform(0.1, 1.0, size=(4, 3)) * rng.choice([-1, 1], size=(4, 3))
    check_input_grad(ReLU(), x)


def test_sigmoid_grad():
    check_input_grad(Sigmoid(), rng.normal(size=(4, 3)))


def test_tanh_grad():
    check_input_grad(Tanh(), rng.normal(size=(4, 3)))
