"""Numerical gradient check: your best friend for catching backprop bugs.

Idea: the derivative is "how much does the output change if I nudge the input".
We nudge each number by a tiny eps, measure the change, and compare that with
the gradient your backward() computed. If they match, backward() is correct.

    f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)     (central difference)
"""

import numpy as np


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gradient of scalar function f() w.r.t. array x, by finite differences.

    f takes no arguments and reads x (which we modify in place, then restore).
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """< 1e-6 is great, < 1e-4 is fine, > 1e-2 means there is a bug."""
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))
