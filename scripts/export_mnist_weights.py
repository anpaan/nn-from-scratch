"""Export the trained MNIST weights to JSON for the browser playground (M9).

Run from the repo root, after training:
    python -m examples.mnist                  # creates data/mnist_weights.npz
    python -m scripts.export_mnist_weights    # creates playground/public/mnist.json
"""

import json
import os

import numpy as np

SRC = os.path.join("data", "mnist_weights.npz")
DST = os.path.join("playground", "public", "mnist.json")
DECIMALS = 4  # 4 decimals keeps the file small (~1 MB) and changes accuracy by ~0


def accuracy(w, X, y):
    hidden = np.maximum(X @ w["W1"] + w["b1"], 0.0)  # Linear + ReLU
    return float(np.mean((hidden @ w["W2"] + w["b2"]).argmax(axis=1) == y))


def export(src: str = SRC, dst: str = DST, decimals: int = DECIMALS, check: bool = True) -> dict:
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found. Train first:  python -m examples.mnist")
    with np.load(src) as d:
        weights = {k: d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2")}
    rounded = {k: np.round(v, decimals) for k, v in weights.items()}

    # Sanity check: rounding must not hurt accuracy (only if the dataset is available).
    if check and os.path.exists(os.path.join("data", "mnist.npz")):
        from engine.data import load_mnist

        _, _, X_test, y_test = load_mnist()
        print(f"test acc, original weights: {accuracy(weights, X_test, y_test):.2%}")
        print(f"test acc, rounded weights:  {accuracy(rounded, X_test, y_test):.2%}")

    out = {
        "layers": [int(weights["W1"].shape[0]), int(weights["W1"].shape[1]), int(weights["W2"].shape[1])],
        # matrices are flattened row by row: W1[i, j] is at index i * 128 + j
        **{k: v.ravel().tolist() for k, v in rounded.items()},
    }
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"saved {dst} ({os.path.getsize(dst) / 1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    export()
