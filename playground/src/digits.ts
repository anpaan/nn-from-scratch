// M9: draw a digit, the MNIST network (trained in Python, exported to public/mnist.json) guesses it.
//
// The network only knows what MNIST digits look like: white strokes on black, about 20x20 px,
// centred by their centre of mass inside a 28x28 image. So before predicting we turn the
// drawing into exactly that format (the small preview shows what the network really sees).
import { Linear, ReLU, Sequential, softmax, type Matrix } from "./engine.ts";

interface MnistWeights {
  layers: [number, number, number];
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
}

/** Copy a flat row-major array into an existing (rows x cols) matrix, in place. */
function fill(target: Matrix, flat: number[]) {
  const cols = target[0].length;
  target.forEach((row, i) => row.forEach((_, j) => (row[j] = flat[i * cols + j])));
}

async function loadModel(): Promise<Sequential> {
  // relative URL: works on localhost and on GitHub Pages (/nn-from-scratch/mnist.json)
  const res = await fetch("mnist.json");
  if (!res.ok) throw new Error(`mnist.json: HTTP ${res.status}`);
  const w = (await res.json()) as MnistWeights;
  const [nIn, nHidden, nOut] = w.layers;
  const l1 = new Linear(nIn, nHidden);
  const l2 = new Linear(nHidden, nOut);
  fill(l1.W, w.W1);
  fill(l1.b, w.b1);
  fill(l2.W, w.W2);
  fill(l2.b, w.b2);
  return new Sequential(l1, new ReLU(), l2);
}

/** Drawing -> 784 numbers in [0, 1], MNIST style. Returns null if the canvas is empty. */
function toMnistInput(draw: HTMLCanvasElement, preview: HTMLCanvasElement): number[] | null {
  const src = draw.getContext("2d")!.getImageData(0, 0, draw.width, draw.height);

  // 1. bounding box of the ink
  let minX = draw.width, minY = draw.height, maxX = -1, maxY = -1;
  for (let y = 0; y < draw.height; y++) {
    for (let x = 0; x < draw.width; x++) {
      if (src.data[(y * draw.width + x) * 4] > 20) {
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) return null;

  // 2. scale the box so its longer side is 20 px, draw it in the middle of a 28x28 canvas
  const w = maxX - minX + 1, h = maxY - minY + 1;
  const scale = 20 / Math.max(w, h);
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d")!;
  sctx.imageSmoothingEnabled = true; // anti-aliased edges, like MNIST
  sctx.imageSmoothingQuality = "high";
  const dw = w * scale, dh = h * scale;
  sctx.drawImage(draw, minX, minY, w, h, (28 - dw) / 2, (28 - dh) / 2, dw, dh);

  // 3. shift so the centre of mass sits at (14, 14)
  let pixels = sctx.getImageData(0, 0, 28, 28).data;
  let sum = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = pixels[i * 4];
    sum += v;
    cx += v * (i % 28);
    cy += v * Math.floor(i / 28);
  }
  const dx = Math.round(14 - cx / sum - 0.5), dy = Math.round(14 - cy / sum - 0.5);
  const shifted = document.createElement("canvas");
  shifted.width = shifted.height = 28;
  const shctx = shifted.getContext("2d")!;
  shctx.fillStyle = "black";
  shctx.fillRect(0, 0, 28, 28);
  shctx.drawImage(small, dx, dy);
  pixels = shctx.getImageData(0, 0, 28, 28).data;

  // 4. show what the network sees, enlarged with hard pixels
  const pctx = preview.getContext("2d")!;
  pctx.imageSmoothingEnabled = false;
  pctx.drawImage(shifted, 0, 0, preview.width, preview.height);

  // 5. 784 numbers in [0, 1], row by row (same as load_mnist in Python)
  const input: number[] = [];
  for (let i = 0; i < 784; i++) input.push(pixels[i * 4] / 255);
  return input;
}

export async function setupDigits() {
  const draw = document.getElementById("digit-canvas") as HTMLCanvasElement;
  const preview = document.getElementById("digit-preview") as HTMLCanvasElement;
  const bars = document.getElementById("digit-bars") as HTMLDivElement;
  const result = document.getElementById("digit-result") as HTMLParagraphElement;
  const clear = document.getElementById("digit-clear") as HTMLButtonElement;
  const ctx = draw.getContext("2d")!;

  // 10 probability bars, created once
  const fills: HTMLDivElement[] = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span>${d}</span><div class="bar"><div class="fill"></div></div><span class="pct"></span>`;
    bars.appendChild(row);
    fills.push(row.querySelector(".fill") as HTMLDivElement);
  }
  const pcts = Array.from(bars.querySelectorAll<HTMLSpanElement>(".pct"));

  function reset() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, draw.width, draw.height);
    const pctx = preview.getContext("2d")!;
    pctx.fillStyle = "black";
    pctx.fillRect(0, 0, preview.width, preview.height);
    fills.forEach((f) => (f.style.width = "0%"));
    pcts.forEach((p) => (p.textContent = ""));
    result.textContent = "Draw a digit (0–9) in the black box.";
  }
  reset();

  let model: Sequential;
  try {
    model = await loadModel();
  } catch (e) {
    result.textContent = "Could not load mnist.json. Run: python -m scripts.export_mnist_weights";
    console.error(e);
    return;
  }

  function predict() {
    const input = toMnistInput(draw, preview);
    if (!input) return;
    const probs = softmax(model.forward([input]))[0];
    const best = probs.indexOf(Math.max(...probs));
    probs.forEach((p, d) => {
      fills[d].style.width = `${(p * 100).toFixed(1)}%`;
      fills[d].classList.toggle("best", d === best);
      pcts[d].textContent = `${(p * 100).toFixed(1)}%`;
    });
    result.textContent = `I think it's a ${best} (${(probs[best] * 100).toFixed(1)}% sure)`;
  }

  // drawing with mouse, pen or finger
  let drawing = false;
  let last: [number, number] | null = null;
  const point = (e: PointerEvent): [number, number] => {
    const r = draw.getBoundingClientRect();
    return [((e.clientX - r.left) * draw.width) / r.width, ((e.clientY - r.top) * draw.height) / r.height];
  };
  draw.addEventListener("pointerdown", (e) => {
    drawing = true;
    last = point(e);
    draw.setPointerCapture(e.pointerId);
  });
  draw.addEventListener("pointermove", (e) => {
    if (!drawing || !last) return;
    const p = point(e);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(last[0], last[1]);
    ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    last = p;
    predict(); // live: the guess updates while you draw
  });
  const stop = () => {
    drawing = false;
    last = null;
  };
  draw.addEventListener("pointerup", stop);
  draw.addEventListener("pointercancel", stop);
  clear.addEventListener("click", reset);
}
