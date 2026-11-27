"""Loss functions: one number that says how wrong the network is.

Each loss has:
  forward(pred, target) -> float      (the loss value)
  backward()            -> np.ndarray (dLoss/dPred, same shape as pred)
"""

import numpy as np


class MSE:
    """Mean squared error, for regression: mean((pred - target)^2)."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target              # cache: backward only needs the difference
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        # d/dp of mean((p - t)^2) = 2 * (p - t) / N, N = total number of elements
        return 2.0 * self._diff / self._diff.size


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy combined, for classification.

    pred   : raw scores ("logits"), shape (batch, n_classes)
    target : integer class labels, shape (batch,)

    Combining the two makes the gradient beautifully simple and numerically
    stable. See TASKS.md M4.
    """

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        # TODO (M4):
        # 1. subtract the row max from logits (stability trick)
        # 2. softmax -> probabilities, cache them
        # 3. loss = mean of -log(prob of the correct class)
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        # TODO (M4): (probs - one_hot(target)) / batch_size
        raise NotImplementedError
