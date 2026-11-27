"""Sequential: chains layers together."""

import numpy as np

from engine.layers import Layer


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def __call__(self, x):
        return self.forward(x)
