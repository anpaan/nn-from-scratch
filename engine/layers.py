"""Layers: the building blocks of a network.

Every layer does two things:
  forward(x)      -> compute the output from the input
  backward(grad)  -> given dLoss/dOutput, compute dLoss/dInput
                     (and store dLoss/dParam for each parameter)

Shapes convention used everywhere in this repo:
  x has shape (batch_size, n_features)   -> one sample per ROW
"""

import numpy as np


class Layer:
    """Base class. Layers without parameters (e.g. ReLU) just inherit this."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[np.ndarray]:
        """Trainable parameters (same order as grads())."""
        return []

    def grads(self) -> list[np.ndarray]:
        """Gradients of the loss w.r.t. each parameter."""
        return []


class Linear(Layer):
    """Fully-connected layer: y = x @ W + b

    W: (in_features, out_features)
    b: (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He initialisation: keeps signals from exploding/vanishing with ReLU.
        # See TASKS.md M1 for why we don't just use zeros.
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_features), size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None  # cache of the input, needed in backward()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO (M1): save x in self._x, then return x @ W + b
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # TODO (M1): grad_out has shape (batch, out_features)
        #   self.dW = ?   shape (in_features, out_features)
        #   self.db = ?   shape (out_features,)
        #   return  ?     shape (batch, in_features)  -> dLoss/dx
        # Hint: check the shapes, there is only one way to make them fit.
        # ⚠ Write INTO the existing arrays: self.dW[...] = ...  (not self.dW = ...)
        #   The optimizer holds references to these exact arrays (see TASKS.md M2).
        raise NotImplementedError

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
