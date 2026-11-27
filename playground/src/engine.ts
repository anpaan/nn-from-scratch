// M7: the same engine as engine/ in Python, re-written in TypeScript.
// Keep it small: plain number[][] arrays are fast enough for 2D toy data.

export type Matrix = number[][];

export interface Layer {
  forward(x: Matrix): Matrix;
  backward(gradOut: Matrix): Matrix;
  params(): Matrix[];
  grads(): Matrix[];
}

// TODO (M7): small matrix helpers: matmul, transpose, add bias, zeros, randn

// TODO (M7): export class Linear implements Layer { ... }
// TODO (M7): export class Tanh / ReLU implements Layer { ... }
// TODO (M7): export class SoftmaxCrossEntropy { forward(...); backward(); }
// TODO (M7): export class Sequential { forward; backward; params; grads }
// TODO (M7): export class SGD { step() }
