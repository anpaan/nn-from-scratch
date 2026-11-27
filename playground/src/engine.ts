// M7: the same engine as engine/ in Python, re-written in TypeScript.
// Keep it small: plain number[][] arrays are fast enough for 2D toy data.
//
// Same conventions as the Python engine:
//   x is (batch, features), one sample per row.  Linear: y = x @ W + b, W is (in, out).
//   backward() writes gradients INTO the existing arrays, and optimizers update
//   parameters IN PLACE, because the optimizer keeps references to those arrays.

export type Matrix = number[][];

export interface Layer {
  forward(x: Matrix): Matrix;
  backward(gradOut: Matrix): Matrix;
  params(): Matrix[];
  grads(): Matrix[];
}

// ---------- small helpers ----------

/** Seeded random generator (mulberry32), so runs are reproducible like seed= in NumPy. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample (Box-Muller). */
export function randn(rng: () => number): number {
  const u = 1 - rng(); // avoid log(0)
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

/** (n, k) @ (k, m) -> (n, m) */
export function matmul(a: Matrix, b: Matrix): Matrix {
  const n = a.length, k = b.length, m = b[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const ai = a[i], oi = out[i];
    for (let p = 0; p < k; p++) {
      const aip = ai[p], bp = b[p];
      for (let j = 0; j < m; j++) oi[j] += aip * bp[j];
    }
  }
  return out;
}

export function transpose(a: Matrix): Matrix {
  const out = zeros(a[0].length, a.length);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
  return out;
}

/** Copy src into dst element by element (dst keeps its identity). Like dW[...] = src in NumPy. */
function copyInto(dst: Matrix, src: Matrix): void {
  for (let i = 0; i < dst.length; i++) for (let j = 0; j < dst[i].length; j++) dst[i][j] = src[i][j];
}

// ---------- layers ----------

export class Linear implements Layer {
  W: Matrix;
  b: Matrix; // shape (1, out): a matrix with one row, so params() can return Matrix[]
  dW: Matrix;
  db: Matrix;
  private x: Matrix = [];

  constructor(inFeatures: number, outFeatures: number, seed = 0) {
    const rng = makeRng(seed);
    const std = Math.sqrt(2 / inFeatures); // He init, same as Python
    this.W = Array.from({ length: inFeatures }, () =>
      Array.from({ length: outFeatures }, () => randn(rng) * std),
    );
    this.b = zeros(1, outFeatures);
    this.dW = zeros(inFeatures, outFeatures);
    this.db = zeros(1, outFeatures);
  }

  forward(x: Matrix): Matrix {
    this.x = x; // cache for backward
    const out = matmul(x, this.W);
    const b = this.b[0];
    for (const row of out) for (let j = 0; j < row.length; j++) row[j] += b[j];
    return out;
  }

  backward(gradOut: Matrix): Matrix {
    copyInto(this.dW, matmul(transpose(this.x), gradOut)); // dW = x.T @ grad_out
    const db = this.db[0];
    db.fill(0);
    for (const row of gradOut) for (let j = 0; j < row.length; j++) db[j] += row[j]; // sum over batch
    return matmul(gradOut, transpose(this.W)); // dx = grad_out @ W.T
  }

  params(): Matrix[] { return [this.W, this.b]; }
  grads(): Matrix[] { return [this.dW, this.db]; }
}

export class Tanh implements Layer {
  private out: Matrix = [];
  forward(x: Matrix): Matrix {
    this.out = x.map((row) => row.map(Math.tanh));
    return this.out;
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.map((row, i) => row.map((g, j) => g * (1 - this.out[i][j] ** 2)));
  }
  params(): Matrix[] { return []; }
  grads(): Matrix[] { return []; }
}

export class Sigmoid implements Layer {
  private out: Matrix = [];
  forward(x: Matrix): Matrix {
    this.out = x.map((row) => row.map((v) => 1 / (1 + Math.exp(-v))));
    return this.out;
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.map((row, i) => row.map((g, j) => g * this.out[i][j] * (1 - this.out[i][j])));
  }
  params(): Matrix[] { return []; }
  grads(): Matrix[] { return []; }
}

export class ReLU implements Layer {
  private x: Matrix = [];
  forward(x: Matrix): Matrix {
    this.x = x;
    return x.map((row) => row.map((v) => (v > 0 ? v : 0)));
  }
  backward(gradOut: Matrix): Matrix {
    return gradOut.map((row, i) => row.map((g, j) => (this.x[i][j] > 0 ? g : 0)));
  }
  params(): Matrix[] { return []; }
  grads(): Matrix[] { return []; }
}

// ---------- loss ----------

/** Row-wise softmax with the max-subtraction trick (no overflow). */
export function softmax(logits: Matrix): Matrix {
  return logits.map((row) => {
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sum = exps.reduce((s, v) => s + v, 0);
    return exps.map((v) => v / sum);
  });
}

export class SoftmaxCrossEntropy {
  private probs: Matrix = [];
  private target: number[] = [];

  forward(logits: Matrix, target: number[]): number {
    this.probs = softmax(logits);
    this.target = target;
    let loss = 0;
    for (let i = 0; i < target.length; i++) loss -= Math.log(this.probs[i][target[i]] + 1e-12);
    return loss / target.length;
  }

  backward(): Matrix {
    const batch = this.target.length;
    // new arrays (map), so the cached probs are not modified: same role as .copy() in Python
    return this.probs.map((row, i) =>
      row.map((p, j) => (p - (j === this.target[i] ? 1 : 0)) / batch),
    );
  }
}

// ---------- network + optimizer ----------

export class Sequential {
  layers: Layer[];
  constructor(...layers: Layer[]) {
    this.layers = layers;
  }
  forward(x: Matrix): Matrix {
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }
  backward(grad: Matrix): Matrix {
    for (let i = this.layers.length - 1; i >= 0; i--) grad = this.layers[i].backward(grad);
    return grad;
  }
  params(): Matrix[] { return this.layers.flatMap((l) => l.params()); }
  grads(): Matrix[] { return this.layers.flatMap((l) => l.grads()); }
}

export class SGD {
  params: Matrix[];
  grads: Matrix[];
  lr: number;
  constructor(params: Matrix[], grads: Matrix[], lr = 0.1) {
    this.params = params;
    this.grads = grads;
    this.lr = lr;
  }
  step(): void {
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k], g = this.grads[k];
      // IN PLACE, element by element: the layers hold these same arrays
      for (let i = 0; i < p.length; i++) for (let j = 0; j < p[i].length; j++) p[i][j] -= this.lr * g[i][j];
    }
  }
}
