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
        shifted = logits - logits.max(axis=1, keepdims=True)   # stability: largest score becomes 0, exp() can't overflow
        exp = np.exp(shifted)
        self._probs = exp / exp.sum(axis=1, keepdims=True)      # softmax: each row sums to 1
        self._target = target
        batch = logits.shape[0]
        correct = self._probs[np.arange(batch), target]         # prob given to the right class, one per row
        return float(-np.mean(np.log(correct + 1e-12)))         # 1e-12 avoids log(0)

    def backward(self) -> np.ndarray:
        batch = self._probs.shape[0]
        grad = self._probs.copy()                     # copy: don't modify the cached probabilities
        grad[np.arange(batch), self._target] -= 1.0   # probs - one_hot(target)
        return grad / batch
