// 2D toy datasets, ported from engine/data.py (same shapes, own random generator).
import { makeRng, randn, type Matrix } from "./engine.ts";

export interface Dataset {
  X: Matrix;
  y: number[];
  nClasses: number;
}

/** Interleaved spiral arms, one arm per class. */
export function makeSpiral(nPerClass = 100, nClasses = 3, noise = 0.2, seed = 0): Dataset {
  const rng = makeRng(seed);
  const X: Matrix = [];
  const y: number[] = [];
  for (let c = 0; c < nClasses; c++) {
    for (let i = 0; i < nPerClass; i++) {
      const r = i / (nPerClass - 1);
      const t = c * 4 + (4 * i) / (nPerClass - 1) + randn(rng) * noise;
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      y.push(c);
    }
  }
  return { X, y, nClasses };
}

/** Inner circle = class 0, outer ring = class 1. */
export function makeCircles(n = 200, noise = 0.05, seed = 0): Dataset {
  const rng = makeRng(seed);
  const X: Matrix = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = rng() * 2 * Math.PI;
    const r = i < n / 2 ? 0.4 : 1.0;
    X.push([r * Math.cos(a) + randn(rng) * noise, r * Math.sin(a) + randn(rng) * noise]);
    y.push(i < n / 2 ? 0 : 1);
  }
  return { X, y, nClasses: 2 };
}

/** Noisy XOR: 4 clusters at the corners, opposite corners share a class.
 *  (The Python version has only the 4 exact points; clusters look better on a heatmap.) */
export function makeXor(nPerCluster = 50, noise = 0.15, seed = 0): Dataset {
  const rng = makeRng(seed);
  const corners: [number, number, number][] = [
    [-0.6, -0.6, 0], [0.6, 0.6, 0], [-0.6, 0.6, 1], [0.6, -0.6, 1],
  ];
  const X: Matrix = [];
  const y: number[] = [];
  for (const [cx, cy, label] of corners) {
    for (let i = 0; i < nPerCluster; i++) {
      X.push([cx + randn(rng) * noise, cy + randn(rng) * noise]);
      y.push(label);
    }
  }
  return { X, y, nClasses: 2 };
}
