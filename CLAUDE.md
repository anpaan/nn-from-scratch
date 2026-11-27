# CLAUDE.md — nn-from-scratch

Learning project by Hoàng Ân (embedded engineer: C, STM32, Raspberry Pi, CAN) to learn ML.
Goal: a neural network engine from scratch in Python + NumPy (no PyTorch/TensorFlow),
then a browser playground in TypeScript + Vite, deployed to GitHub Pages.

## Language
- Reply in Vietnamese when Ân writes Vietnamese. Keep technical terms in English
  (add the Vietnamese meaning the first time a term appears).
- Code, comments, commit messages: English. `TASKS.md` and `docs/notes.md`: Vietnamese.

## Teaching mode (most important rule)
The `TODO` parts of `engine/` and `examples/` are Ân's exercises. **Do not implement them**
unless he explicitly says "cho xem code" / "viết luôn" for that specific part.
- Default: explain the idea, then hint in steps (guiding question → shape/formula → pseudo-code).
- When he asks for a review: say what is right/wrong and *why*; point out silent bugs
  (shapes, broadcasting, in-place vs. new array, overflow). Suggest the fix, don't rewrite the file.
- You MAY write freely: UI boilerplate, CI, data-download scripts, plotting helpers, README, tooling.
- Do not edit or weaken tests to make them pass.
- If unsure about an API or a number, say so. Don't invent.

## Code conventions
- Shapes: `x` is `(batch, features)`, one sample per row. `Linear`: `y = x @ W + b`, `W (in, out)`.
- Every layer has `forward`, `backward`, `params()`, `grads()`.
- `backward` writes gradients **into existing arrays** (`self.dW[...] = ...`), because the
  optimizer keeps references. Assigning a new array makes the net silently not learn
  while gradient tests still pass.
- Optimizers update **in-place** (`p -= ...`), never `p = p - ...`.
- Softmax subtracts the row max before `exp` (overflow).

## Commands (Windows, PowerShell)
- Activate venv: `.venv\Scripts\Activate.ps1`
- Tests: `python -m pytest -v` from repo root. `NotImplementedError` → SKIPPED (via `conftest.py`).
  SKIPPED = not written yet, FAILED = wrong, PASSED = done.
- Run examples from repo root as modules: `python -m examples.xor`
- Playground: `cd playground; npm install; npm run dev`

## Roadmap
See `TASKS.md` (milestones M0–M10) and `docs/notes.md` (concept notes).
