"""MNIST loader test. Skipped when data/mnist.npz is not present (e.g. on CI)."""

import os

import numpy as np
import pytest

from engine.data import load_mnist

PATH = "data/mnist.npz"


@pytest.mark.skipif(not os.path.exists(PATH), reason="data/mnist.npz not downloaded")
def test_load_mnist_shapes_and_range():
    X_train, y_train, X_test, y_test = load_mnist(PATH)
    assert X_train.shape == (60000, 784)
    assert X_test.shape == (10000, 784)
    assert y_train.shape == (60000,) and y_test.shape == (10000,)
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert set(np.unique(y_train)) == set(range(10))


def test_load_mnist_missing_file_message():
    with pytest.raises(FileNotFoundError, match="download_mnist"):
        load_mnist("data/does_not_exist.npz")
