"""Optimizer tests: update rules and in-place behaviour."""

import numpy as np

from engine import Linear, Tanh, Sequential, SoftmaxCrossEntropy, SGD, Adam
from engine.data import make_xor


def test_sgd_step_in_place():
    p, g = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    ref = p                      # same array object
    SGD([p], [g], lr=0.1).step()
    assert ref is p
    assert np.allclose(p, [0.95, 2.1])


def test_adam_first_step():
    # After bias correction, the first Adam step is lr * g/|g| = lr * sign(g)
    p, g = np.array([1.0, 1.0]), np.array([0.5, -3.0])
    ref = p
    Adam([p], [g], lr=0.01).step()
    assert ref is p
    assert np.allclose(p, [0.99, 1.01], atol=1e-6)


def test_adam_learns_xor():
    X, y = make_xor()
    model = Sequential(Linear(2, 8, seed=1), Tanh(), Linear(8, 2, seed=2))
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model.params(), model.grads(), lr=0.05)
    for _ in range(500):
        loss_fn.forward(model(X), y)
        model.backward(loss_fn.backward())
        opt.step()
    assert np.all(model(X).argmax(axis=1) == y)
