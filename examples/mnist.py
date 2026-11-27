"""M6: handwritten digit recognition (target: ~97% test accuracy).

Run from the repo root:
    python -m scripts.download_mnist      # once
    python -m examples.mnist
"""

import argparse
import time

import numpy as np

from engine import Linear, ReLU, Sequential, SoftmaxCrossEntropy, Adam
from engine.data import load_mnist

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

X_train, y_train, X_test, y_test = load_mnist()
print(f"train {X_train.shape}, test {X_test.shape}")

model = Sequential(Linear(784, 128, seed=1), ReLU(), Linear(128, 10, seed=2))
loss_fn = SoftmaxCrossEntropy()
opt = Adam(model.params(), model.grads(), lr=args.lr)
rng = np.random.default_rng(0)


def accuracy(X, y):
    return np.mean(model(X).argmax(axis=1) == y)


n = len(X_train)
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    order = rng.permutation(n)          # shuffle every epoch: batches differ each time
    total_loss = 0.0
    for start in range(0, n, args.batch):
        idx = order[start:start + args.batch]
        xb, yb = X_train[idx], y_train[idx]
        loss = loss_fn.forward(model(xb), yb)     # 1-2. forward + loss on ONE mini-batch
        model.backward(loss_fn.backward())        # 3. backward
        opt.step()                                # 4. update after every batch
        total_loss += loss * len(idx)
    print(
        f"epoch {epoch}  train loss {total_loss / n:.4f}  "
        f"train acc {accuracy(X_train, y_train):.2%}  test acc {accuracy(X_test, y_test):.2%}  "
        f"({time.time() - t0:.1f}s)"
    )

# Save weights for the browser playground (M9). data/ is git-ignored.
W1, b1, W2, b2 = model.params()
np.savez("data/mnist_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("saved data/mnist_weights.npz")
