"""Test the MNIST weight export with small fake weights (no dataset needed)."""

import json

import numpy as np

from scripts.export_mnist_weights import export


def test_export_shapes_and_rounding(tmp_path):
    rng = np.random.default_rng(0)
    src, dst = tmp_path / "w.npz", tmp_path / "out" / "mnist.json"
    np.savez(src, W1=rng.normal(size=(784, 128)), b1=rng.normal(size=128),
             W2=rng.normal(size=(128, 10)), b2=rng.normal(size=10))
    export(str(src), str(dst), decimals=4, check=False)
    data = json.loads(dst.read_text())
    assert data["layers"] == [784, 128, 10]
    assert len(data["W1"]) == 784 * 128 and len(data["b1"]) == 128
    assert len(data["W2"]) == 128 * 10 and len(data["b2"]) == 10
    w1 = np.load(src)["W1"]
    assert data["W1"][5 * 128 + 7] == round(float(w1[5, 7]), 4)  # row-major order
