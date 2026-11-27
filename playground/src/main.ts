// Entry point of the playground UI.
//
// The core idea is a loop driven by requestAnimationFrame (about 60 times/s):
//   each frame: run a few training steps -> redraw boundary, network, loss chart
// That's what makes the training look "live".
//
// M8: the controls on the left rebuild the network; changing any setting resets training.

import { Sequential, Linear, Tanh, ReLU, Sigmoid, SoftmaxCrossEntropy, SGD, type Layer } from "./engine.ts";
import { makeSpiral, makeCircles, makeXor, type Dataset } from "./data.ts";
import { drawBoundary } from "./boundary.ts";
import { NetworkView } from "./network-view.ts";
import { drawLossChart } from "./loss-chart.ts";

// ---------- DOM ----------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  dataset: $<HTMLSelectElement>("dataset"),
  lr: $<HTMLSelectElement>("lr"),
  layers: $<HTMLSelectElement>("layers"),
  neurons: $<HTMLSelectElement>("neurons"),
  activation: $<HTMLSelectElement>("activation"),
  play: $<HTMLButtonElement>("play"),
  reset: $<HTMLButtonElement>("reset"),
  boundary: $<HTMLCanvasElement>("boundary"),
  lossChart: $<HTMLCanvasElement>("loss-chart"),
  status: $<HTMLParagraphElement>("status"),
};
const netView = new NetworkView(document.getElementById("net-svg") as unknown as SVGSVGElement);

// ---------- training state ----------

const DATASETS: Record<string, () => Dataset> = {
  spiral: () => makeSpiral(),
  circles: () => makeCircles(),
  xor: () => makeXor(),
};
const ACTIVATIONS: Record<string, () => Layer> = {
  tanh: () => new Tanh(),
  relu: () => new ReLU(),
  sigmoid: () => new Sigmoid(),
};

const STEPS_PER_FRAME = 2;   // training steps between two redraws
const MAX_EPOCHS = 5000;     // auto-pause afterwards so the page doesn't burn CPU forever

let data: Dataset;
let model: Sequential;
let opt: SGD;
const lossFn = new SoftmaxCrossEntropy();
let epoch = 0;
let loss = 0;
let lossHistory: number[] = [];
let running = true;

/** (Re)build dataset, network and optimizer from the current control values. */
function rebuild() {
  data = (DATASETS[ui.dataset.value] ?? DATASETS.spiral)();
  const nLayers = Number(ui.layers.value);
  const nNeurons = Number(ui.neurons.value);
  const sizes = [2, ...Array<number>(nLayers).fill(nNeurons), data.nClasses];

  const layers: Layer[] = [];
  const seed = Math.floor(Math.random() * 1e9); // new random init on every reset
  for (let l = 0; l < sizes.length - 1; l++) {
    layers.push(new Linear(sizes[l], sizes[l + 1], seed + l));
    if (l < sizes.length - 2) layers.push(ACTIVATIONS[ui.activation.value]()); // no activation after the last Linear
  }
  model = new Sequential(...layers);
  opt = new SGD(model.params(), model.grads(), Number(ui.lr.value));

  epoch = 0;
  loss = 0;
  lossHistory = [];
  netView.build(model, sizes);
  draw();
}

function accuracy(): number {
  const out = model.forward(data.X);
  let ok = 0;
  out.forEach((row, i) => {
    if (row.indexOf(Math.max(...row)) === data.y[i]) ok++;
  });
  return ok / data.y.length;
}

function draw() {
  drawBoundary(ui.boundary, model, data);
  netView.update();
  drawLossChart(ui.lossChart, lossHistory);
  ui.status.textContent =
    `epoch ${epoch}  |  loss ${loss.toFixed(4)}  |  acc ${(accuracy() * 100).toFixed(1)}%` +
    (epoch >= MAX_EPOCHS ? "  (max epochs reached)" : "");
}

function setRunning(on: boolean) {
  running = on;
  ui.play.textContent = on ? "⏸ Pause" : "▶ Play";
}

// ---------- the live loop ----------

function frame() {
  if (running) {
    for (let s = 0; s < STEPS_PER_FRAME && epoch < MAX_EPOCHS; s++) {
      loss = lossFn.forward(model.forward(data.X), data.y); // 1-2. forward + loss
      model.backward(lossFn.backward());                    // 3. backward
      opt.step();                                           // 4. update
      epoch++;
    }
    lossHistory.push(loss);
    if (epoch >= MAX_EPOCHS) setRunning(false);
    draw();
  }
  requestAnimationFrame(frame);
}

// ---------- wire up the controls ----------

for (const el of [ui.dataset, ui.lr, ui.layers, ui.neurons, ui.activation]) {
  el.addEventListener("change", () => {
    rebuild();
    setRunning(true); // "change a setting -> the network resets and learns again"
  });
}
ui.play.addEventListener("click", () => {
  if (epoch >= MAX_EPOCHS) rebuild();
  setRunning(!running);
});
ui.reset.addEventListener("click", () => {
  rebuild();
  setRunning(true);
});

rebuild();
setRunning(true);
requestAnimationFrame(frame);
