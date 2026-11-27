// Entry point of the playground UI.
//
// The core idea is a loop driven by requestAnimationFrame (about 60 times/s):
//   each frame: run a few training steps -> redraw the decision boundary
// That's what makes the training look "live".
//
// M7: pick the dataset with the URL, e.g.  ?data=spiral  ?data=circles  ?data=xor
// (real controls come in M8).

import { Sequential, Linear, Tanh, SoftmaxCrossEntropy, SGD, softmax, type Matrix } from "./engine.ts";
import { makeSpiral, makeCircles, makeXor, type Dataset } from "./data.ts";

// ---------- setup ----------

const datasets: Record<string, () => Dataset> = {
  spiral: () => makeSpiral(),
  circles: () => makeCircles(),
  xor: () => makeXor(),
};
const name = new URLSearchParams(location.search).get("data") ?? "spiral";
const data = (datasets[name] ?? datasets.spiral)();

const model = new Sequential(
  new Linear(2, 16, 1), new Tanh(),
  new Linear(16, 16, 2), new Tanh(),
  new Linear(16, data.nClasses, 3),
);
const lossFn = new SoftmaxCrossEntropy();
const opt = new SGD(model.params(), model.grads(), 0.3);

const STEPS_PER_FRAME = 2; // training steps between two redraws (higher = faster, less "live")
const MAX_EPOCHS = 4000;   // stop afterwards so the page doesn't burn CPU forever
let epoch = 0;
let loss = 0;

// ---------- drawing ----------

const canvas = document.querySelector<HTMLCanvasElement>("#boundary")!;
const ctx = canvas.getContext("2d")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;

const RANGE = 1.5;  // the canvas shows x and y in [-RANGE, RANGE]
const CELLS = 60;   // heatmap resolution: CELLS x CELLS squares
const COLORS: [number, number, number][] = [
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

function toPixel(x: number, y: number): [number, number] {
  return [((x + RANGE) / (2 * RANGE)) * canvas.width, ((RANGE - y) / (2 * RANGE)) * canvas.height];
}

function draw() {
  // 1. heatmap: ask the network about every grid point, mix class colours by probability
  const probs = softmax(model.forward(grid));
  const cw = canvas.width / CELLS, ch = canvas.height / CELLS;
  probs.forEach((p, k) => {
    let r = 0, g = 0, b = 0;
    p.forEach((pk, cls) => {
      r += pk * COLORS[cls][0];
      g += pk * COLORS[cls][1];
      b += pk * COLORS[cls][2];
    });
    // blend with white so the data points stay readable
    ctx.fillStyle = `rgb(${(r + 255) / 2}, ${(g + 255) / 2}, ${(b + 255) / 2})`;
    const row = Math.floor(k / CELLS), col = k % CELLS;
    ctx.fillRect(col * cw, row * ch, cw + 1, ch + 1);
  });

  // 2. data points on top
  data.X.forEach(([x, y], i) => {
    const [px, py] = toPixel(x, y);
    const [cr, cg, cb] = COLORS[data.y[i]];
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = `rgb(${cr}, ${cg}, ${cb})`;
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.stroke();
  });
}

function accuracy(): number {
  const out = model.forward(data.X);
  let ok = 0;
  out.forEach((row, i) => {
    if (row.indexOf(Math.max(...row)) === data.y[i]) ok++;
  });
  return ok / data.y.length;
}

// ---------- the live loop ----------

function frame() {
  for (let s = 0; s < STEPS_PER_FRAME && epoch < MAX_EPOCHS; s++) {
    loss = lossFn.forward(model.forward(data.X), data.y); // 1-2. forward + loss
    model.backward(lossFn.backward());                    // 3. backward
    opt.step();                                           // 4. update
    epoch++;
  }
  draw();
  const done = epoch >= MAX_EPOCHS ? "  (done)" : "";
  status.textContent =
    `${name}  |  epoch ${epoch}  |  loss ${loss.toFixed(4)}  |  acc ${(accuracy() * 100).toFixed(1)}%${done}`;
  if (epoch < MAX_EPOCHS) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
