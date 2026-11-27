// Tests for the TypeScript engine. Run from playground/:  npm test
// Uses Node's built-in test runner (no extra packages needed).
import { test } from "node:test";
import assert from "node:assert/strict";
import { Linear, Tanh, ReLU, Sequential, SoftmaxCrossEntropy, SGD, type Matrix } from "./engine.ts";
import { makeSpiral } from "./data.ts";

const EPS = 1e-5;

/** Numerical gradient of f() w.r.t. every entry of m (central difference), like engine/gradcheck.py. */
function numericalGrad(f: () => number, m: Matrix): Matrix {
  return m.map((row, i) =>
    row.map((_, j) => {
      const old = m[i][j];
      m[i][j] = old + EPS;
      const plus = f();
      m[i][j] = old - EPS;
      const minus = f();
      m[i][j] = old;
      return (plus - minus) / (2 * EPS);
    }),
  );
}

function assertClose(a: Matrix, b: Matrix, tol = 1e-6) {
  a.forEach((row, i) => row.forEach((v, j) => assert.ok(Math.abs(v - b[i][j]) < tol, `[${i}][${j}] ${v} vs ${b[i][j]}`)));
}

const x: Matrix = [[0.5, -1.2], [0.3, 0.8], [-0.7, 0.1], [1.1, -0.4]];
const y = [0, 2, 1, 2];

test("Linear forward shape", () => {
  const out = new Linear(2, 5, 0).forward(x);
  assert.equal(out.length, 4);
  assert.equal(out[0].length, 5);
});

test("gradients of a full network match numerical gradients", () => {
  const model = new Sequential(new Linear(2, 6, 1), new Tanh(), new Linear(6, 6, 2), new ReLU(), new Linear(6, 3, 3));
  const lossFn = new SoftmaxCrossEntropy();
  const f = () => lossFn.forward(model.forward(x), y);
  f();
  model.backward(lossFn.backward());
  const grads = model.grads().map((g) => g.map((row) => [...row])); // snapshot
  model.params().forEach((p, k) => assertClose(grads[k], numericalGrad(f, p)));
});

test("softmax cross-entropy is stable with huge logits", () => {
  const loss = new SoftmaxCrossEntropy().forward([[1000, 0]], [0]);
  assert.ok(Number.isFinite(loss));
});

test("SGD updates parameters in place", () => {
  const layer = new Linear(2, 3, 0);
  const W = layer.W;
  layer.forward(x.map((r) => r.slice()).slice(0, 3));
  layer.backward([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const before = W[0][0];
  new SGD(layer.params(), layer.grads(), 0.1).step();
  assert.equal(layer.W, W); // same array object
  assert.notEqual(W[0][0], before);
});

test("network learns the spiral (> 90% accuracy)", () => {
  const data = makeSpiral();
  const model = new Sequential(new Linear(2, 16, 1), new Tanh(), new Linear(16, 16, 2), new Tanh(), new Linear(16, 3, 3));
  const lossFn = new SoftmaxCrossEntropy();
  const opt = new SGD(model.params(), model.grads(), 0.3);
  for (let e = 0; e < 2000; e++) {
    lossFn.forward(model.forward(data.X), data.y);
    model.backward(lossFn.backward());
    opt.step();
  }
  const out = model.forward(data.X);
  const acc = out.filter((row, i) => row.indexOf(Math.max(...row)) === data.y[i]).length / data.y.length;
  assert.ok(acc > 0.9, `accuracy ${acc}`);
});
