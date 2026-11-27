"""M5: classify the 3-arm spiral and plot the decision boundary.

Run from the repo root:  python -m examples.spiral
"""

import numpy as np
import matplotlib.pyplot as plt

from engine import Linear, ReLU, Sequential, SoftmaxCrossEntropy, Adam
from engine.data import make_spiral

X, y = make_spiral()

# TODO (M5): build a model (try 2 -> 64 -> 64 -> 3 with ReLU), an Adam optimizer,
#            and a training loop like examples/xor.py. Print loss and accuracy.
model = None

# Decision boundary: ask the model about every point of a grid and colour it.
def plot_boundary(model, X, y, path="docs/spiral.png"):
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model(grid).argmax(axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="brg")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="brg")
    plt.title("Spiral decision boundary")
    plt.savefig(path, dpi=120)
    print("saved", path)

if model is not None:
    plot_boundary(model, X, y)
