"""M5: classify the 3-arm spiral and plot the decision boundary.

Run from the repo root:
    python -m examples.spiral                 # Adam (default)
    python -m examples.spiral --opt sgd       # compare with plain SGD
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save to file only, no window needed
import matplotlib.pyplot as plt

from engine import Linear, ReLU, Sequential, SoftmaxCrossEntropy, SGD, Adam
from engine.data import make_spiral

parser = argparse.ArgumentParser()
parser.add_argument("--opt", choices=["adam", "sgd"], default="adam")
parser.add_argument("--lr", type=float, default=None, help="default: 0.01 for adam, 0.1 for sgd")
parser.add_argument("--epochs", type=int, default=2000)
args = parser.parse_args()

X, y = make_spiral()

model = Sequential(
    Linear(2, 64, seed=1), ReLU(),
    Linear(64, 64, seed=2), ReLU(),
    Linear(64, 3, seed=3),
)
loss_fn = SoftmaxCrossEntropy()
if args.opt == "adam":
    opt = Adam(model.params(), model.grads(), lr=args.lr or 0.01)
else:
    opt = SGD(model.params(), model.grads(), lr=args.lr or 0.1)

for epoch in range(args.epochs + 1):
    logits = model(X)                    # 1. forward
    loss = loss_fn.forward(logits, y)    # 2. loss
    model.backward(loss_fn.backward())   # 3. backward
    opt.step()                           # 4. update
    if epoch in (25, 50, 100) or epoch % 200 == 0:
        acc = np.mean(logits.argmax(axis=1) == y)
        print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.1%}")


# Decision boundary: ask the model about every point of a grid and colour it.
def plot_boundary(model, X, y, path):
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model(grid).argmax(axis=1).reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="brg")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="brg")
    plt.title(f"Spiral decision boundary ({args.opt})")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)


plot_boundary(model, X, y, "docs/spiral.png" if args.opt == "adam" else "docs/spiral_sgd.png")
