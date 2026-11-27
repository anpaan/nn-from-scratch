"""End-to-end: can the network actually learn XOR?"""

import numpy as np

from engine import Linear, Tanh, Sequential, SoftmaxCrossEntropy, SGD
from engine.data import make_xor


def test_learns_xor():
    X, y = make_xor()
    model = Sequential(Linear(2, 8, seed=1), Tanh(), Linear(8, 2, seed=2))
    loss_fn = SoftmaxCrossEntropy()
    opt = SGD(model.params(), model.grads(), lr=0.5)
    for _ in range(2000):
        loss_fn.forward(model(X), y)
        model.backward(loss_fn.backward())
        opt.step()
    assert np.all(model(X).argmax(axis=1) == y)
