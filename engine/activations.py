"""Activation functions: add non-linearity between Linear layers.

Without them, stacking Linear layers is still just ONE linear function,
so the network could only draw straight lines. See docs/notes.md section 3.
"""

import numpy as np

from engine.layers import Layer


class ReLU(Layer):
    """ReLU(x) = max(0, x).  Fully implemented as a reference example."""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0          # remember where the input was positive
        return x * self._mask

    def backward(self, grad_out):
        # derivative is 1 where x > 0 and 0 elsewhere -> just pass the gradient through the mask
        return grad_out * self._mask


class Sigmoid(Layer):
    """sigmoid(x) = 1 / (1 + e^-x), squashes values into (0, 1)."""

    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad_out):
        s = self._out
        return grad_out * s * (1.0 - s)


class Tanh(Layer):
    """tanh(x), squashes values into (-1, 1)."""

    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_out):
        return grad_out * (1.0 - self._out ** 2)
