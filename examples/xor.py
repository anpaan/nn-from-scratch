"""M3: train a tiny network on XOR and print the loss going down.

Run from the repo root:  python -m examples.xor
"""

from engine import Linear, Tanh, Sequential, SoftmaxCrossEntropy, SGD
from engine.data import make_xor

X, y = make_xor()
model = Sequential(Linear(2, 8, seed=1), Tanh(), Linear(8, 2, seed=2))
loss_fn = SoftmaxCrossEntropy()
opt = SGD(model.params(), model.grads(), lr=0.5)

for epoch in range(2000):
    # The 4 steps of EVERY training loop, forever:
    logits = model(X)                    # 1. forward
    loss = loss_fn.forward(logits, y)    # 2. loss
    model.backward(loss_fn.backward())   # 3. backward
    opt.step()                           # 4. update
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.4f}")

print("predictions:", model(X).argmax(axis=1), " targets:", y)
