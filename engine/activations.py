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
        # TODO (M1): compute and cache the output
        raise NotImplementedError

    def backward(self, grad_out):
        # TODO (M1): d sigmoid / dx = s * (1 - s), where s is the forward output
        raise NotImplementedError


class Tanh(Layer):
    """tanh(x), squashes values into (-1, 1)."""

    def __init__(self):
        self._out = None

    def forward(self, x):
        # TODO (M1)
        raise NotImplementedError

    def backward(self, grad_out):
        # TODO (M1): d tanh / dx = 1 - tanh(x)^2
        raise NotImplementedError
