"""Optimizers: use the gradients to update the parameters.

They modify the parameter arrays IN PLACE (p -= ...), so the layers
see the new values automatically.
"""

import numpy as np


class SGD:
    """Plain gradient descent: p = p - lr * grad"""

    def __init__(self, params: list[np.ndarray], grads: list[np.ndarray], lr: float = 0.1):
        self.params = params
        self.grads = grads
        self.lr = lr

    def step(self):
        # TODO (M2): for each (p, g): update p in place.
        # Careful: `p = p - ...` creates a NEW array and does nothing to the layer!
        raise NotImplementedError


class Adam:
    """Adam: SGD with momentum + per-parameter adaptive step size."""

    def __init__(self, params, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.grads = grads
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]   # running mean of grads
        self.v = [np.zeros_like(p) for p in params]   # running mean of grads^2
        self.t = 0

    def step(self):
        # TODO (M5): see TASKS.md M5 for the formulas
        raise NotImplementedError
