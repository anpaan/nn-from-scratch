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
        for p, g in zip(self.params, self.grads):
            p -= self.lr * g    # IN PLACE: modifies the array the layer holds. Never `p = p - ...`


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
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g           # momentum: running mean of the gradient
            v *= b2
            v += (1 - b2) * g * g       # running mean of gradient^2 (how "loud" this parameter is)
            m_hat = m / (1 - b1 ** self.t)   # bias correction: m and v start at 0
            v_hat = v / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # IN PLACE, like SGD
