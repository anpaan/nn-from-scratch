// Entry point of the playground UI.
//
// The core idea is a loop driven by requestAnimationFrame (about 60 times/s):
//   each frame: run a few training steps -> redraw boundary, network, loss chart
// That's what makes the training look "live".

// import { Sequential, Linear, Tanh, SoftmaxCrossEntropy, SGD } from "./engine";

const boundary = document.querySelector<HTMLCanvasElement>("#boundary")!;
const ctx = boundary.getContext("2d")!;

ctx.font = "14px system-ui";
ctx.fillText("TODO (M7): draw the decision boundary here", 20, 180);

function frame() {
  // TODO (M7): train for N steps, then draw
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
