"""Sequential: chains layers together."""

import numpy as np

from engine.layers import Layer


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO (M2): pass x through each layer in order
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # TODO (M2): pass grad through each layer in REVERSE order
        raise NotImplementedError

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def __call__(self, x):
        return self.forward(x)
