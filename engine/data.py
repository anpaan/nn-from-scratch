"""Toy datasets. These are given, so you can focus on the network itself."""

import numpy as np


def make_xor():
    """The 4 XOR points. Labels: 0 or 1. Impossible with a straight line!"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def make_spiral(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """Interleaved spiral arms, one arm per class. A classic hard 2D dataset."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def make_circles(n: int = 200, noise: float = 0.05, seed: int = 0):
    """Inner circle = class 0, outer ring = class 1."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) < n // 2, 0.4, 1.0)
    X = np.c_[radius * np.cos(angles), radius * np.sin(angles)] + rng.normal(0, noise, (n, 2))
    y = (np.arange(n) >= n // 2).astype(int)
    return X, y


def load_mnist(path: str = "data/mnist.npz"):
    """TODO (M6): load MNIST from a local .npz file.

    Download mnist.npz once (see TASKS.md M6), keep it in data/ (git-ignored).
    Return X_train (60000, 784) scaled to [0, 1], y_train, X_test, y_test.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it first:  python -m scripts.download_mnist"
        )
    with np.load(path) as d:
        # images are uint8 0..255 with shape (N, 28, 28): flatten each to one row of 784
        # and scale to [0, 1]. float32 halves the memory (60000*784 floats).
        X_train = d["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        X_test = d["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = d["y_train"].astype(np.int64)
        y_test = d["y_test"].astype(np.int64)
    return X_train, y_train, X_test, y_test
