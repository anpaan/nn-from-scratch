// Decision-boundary heatmap (moved here from main.ts in M8).
import { softmax, type Matrix, type Sequential } from "./engine.ts";
import type { Dataset } from "./data.ts";

export const RANGE = 1.5; // the canvas shows x and y in [-RANGE, RANGE]
const CELLS = 60;          // heatmap resolution: CELLS x CELLS squares

export const CLASS_COLORS: [number, number, number][] = [
  [235, 120, 60],  // class 0: orange
  [60, 120, 235],  // class 1: blue
  [60, 175, 90],   // class 2: green
];

// Grid of points at the centre of each heatmap cell, built once.
const grid: Matrix = [];
for (let r = 0; r < CELLS; r++) {
  for (let c = 0; c < CELLS; c++) {
    grid.push([
      -RANGE + ((c + 0.5) / CELLS) * 2 * RANGE,
      RANGE - ((r + 0.5) / CELLS) * 2 * RANGE, // canvas y grows downwards
    ]);
  }
}

export function drawBoundary(canvas: HTMLCanvasElement, model: Sequential, data: Dataset) {
  const ctx = canvas.getContext("2d")!;
  const toPixel = (x: number, y: number): [number, number] => [
    ((x + RANGE) / (2 * RANGE)) * canvas.width,
    ((RANGE - y) / (2 * RANGE)) * canvas.height,
  ];

  // 1. heatmap: ask the network about every grid point, mix class colours by probability
  const probs = softmax(model.forward(grid));
  const cw = canvas.width / CELLS, ch = canvas.height / CELLS;
  probs.forEach((p, k) => {
    let r = 0, g = 0, b = 0;
    p.forEach((pk, cls) => {
      r += pk * CLASS_COLORS[cls][0];
      g += pk * CLASS_COLORS[cls][1];
      b += pk * CLASS_COLORS[cls][2];
    });
    // blend with white so the data points stay readable
    ctx.fillStyle = `rgb(${(r + 255) / 2}, ${(g + 255) / 2}, ${(b + 255) / 2})`;
    const row = Math.floor(k / CELLS), col = k % CELLS;
    ctx.fillRect(col * cw, row * ch, cw + 1, ch + 1);
  });

  // 2. data points on top
  data.X.forEach(([x, y], i) => {
    const [px, py] = toPixel(x, y);
    const [cr, cg, cb] = CLASS_COLORS[data.y[i]];
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = `rgb(${cr}, ${cg}, ${cb})`;
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.stroke();
  });
}
