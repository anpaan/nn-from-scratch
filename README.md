# nn-from-scratch

![tests](https://github.com/<your-username>/nn-from-scratch/actions/workflows/ci.yml/badge.svg)

A neural network engine written from scratch with **NumPy only** (no PyTorch, no TensorFlow), plus an interactive **browser playground** where you can watch a network learn in real time.

> 🚧 Work in progress. See [TASKS.md](TASKS.md) for the roadmap.

## What's inside

| Part | What it is |
|---|---|
| `engine/` | Layers, activations, losses and optimizers with hand-written backpropagation |
| `tests/` | Numerical gradient checks for every layer and loss |
| `examples/` | XOR, spiral classification, MNIST digit recognition |
| `playground/` | TypeScript re-implementation + live visualisation (decision boundary, network graph, loss curve) |
| `docs/notes.md` | My study notes on the fundamentals (Vietnamese) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -v          # run the gradient checks
python -m examples.xor       # train your first network
```

Playground:

```bash
cd playground
npm install
npm run dev
```

## Results

_Coming soon: spiral decision boundary, MNIST accuracy, playground demo GIF._

## What I learned

_To be written after v1._

## License

MIT
