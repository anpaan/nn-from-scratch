"""M6: handwritten digit recognition (target: ~97% test accuracy).

Run from the repo root:  python -m examples.mnist
"""

from engine.data import load_mnist

# TODO (M6):
# 1. X_train, y_train, X_test, y_test = load_mnist()
# 2. model: 784 -> 128 -> ReLU -> 10
# 3. MINI-BATCH training: shuffle each epoch, loop over batches of 64
# 4. after each epoch, print test accuracy
# 5. save the weights with np.savez("data/mnist_weights.npz", ...) for the playground (M9)
