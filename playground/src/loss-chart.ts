// M8: loss over time, drawn on a small canvas.
export function drawLossChart(canvas: HTMLCanvasElement, history: number[]) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height, pad = 8, top = 24; // leave room for the labels
  ctx.clearRect(0, 0, W, H);
  if (history.length < 2) return;

  const max = Math.max(...history);
  const x = (k: number) => pad + (k / (history.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - (v / max) * (H - pad - top);

  ctx.beginPath();
  history.forEach((v, k) => (k === 0 ? ctx.moveTo(x(k), y(v)) : ctx.lineTo(x(k), y(v))));
  ctx.strokeStyle = "rgb(60, 120, 235)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = "#555";
  ctx.font = "11px system-ui";
  ctx.fillText(`loss ${history[history.length - 1].toFixed(4)}`, W - 90, 16);
  ctx.fillText(`max ${max.toFixed(2)}`, pad + 2, 16);
}
