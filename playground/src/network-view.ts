// M8: draw the network as SVG: neurons are circles, connections are lines.
// Line thickness follows |w|, colour follows the sign (blue = positive, orange = negative).
// The elements are created once per network (build) and only their attributes change
// every frame (update), which is much cheaper than rebuilding the SVG 60 times a second.
import { Linear, type Sequential } from "./engine.ts";
import { CLASS_COLORS } from "./boundary.ts";

const SVG_NS = "http://www.w3.org/2000/svg";
const POS = "rgb(60, 120, 235)";
const NEG = "rgb(235, 120, 60)";
const R = 11; // neuron radius

export class NetworkView {
  private svg: SVGSVGElement;
  private lines: { el: SVGLineElement; layer: Linear; i: number; j: number }[] = [];

  constructor(svg: SVGSVGElement) {
    this.svg = svg;
  }

  /** Create circles and lines for a network. sizes = neurons per layer, e.g. [2, 8, 8, 3]. */
  build(model: Sequential, sizes: number[]) {
    this.svg.replaceChildren();
    this.lines = [];
    const W = this.svg.clientWidth || Number(this.svg.getAttribute("width"));
    const H = this.svg.clientHeight || Number(this.svg.getAttribute("height"));
    const marginX = 40, marginY = 30;

    // (x, y) of every neuron, layer by layer
    const pos = sizes.map((n, l) => {
      const x = marginX + (l * (W - 2 * marginX)) / (sizes.length - 1);
      const gap = Math.min(40, (H - 2 * marginY) / Math.max(n - 1, 1));
      const top = H / 2 - (gap * (n - 1)) / 2;
      return Array.from({ length: n }, (_, i) => [x, top + i * gap] as [number, number]);
    });

    // lines first, so circles are drawn on top of them
    const linears = model.layers.filter((l): l is Linear => l instanceof Linear);
    linears.forEach((layer, l) => {
      for (let i = 0; i < sizes[l]; i++) {
        for (let j = 0; j < sizes[l + 1]; j++) {
          const el = document.createElementNS(SVG_NS, "line");
          el.setAttribute("x1", String(pos[l][i][0]));
          el.setAttribute("y1", String(pos[l][i][1]));
          el.setAttribute("x2", String(pos[l + 1][j][0]));
          el.setAttribute("y2", String(pos[l + 1][j][1]));
          this.svg.appendChild(el);
          this.lines.push({ el, layer, i, j });
        }
      }
    });

    // circles + labels
    pos.forEach((layerPos, l) => {
      layerPos.forEach(([x, y], i) => {
        const c = document.createElementNS(SVG_NS, "circle");
        c.setAttribute("cx", String(x));
        c.setAttribute("cy", String(y));
        c.setAttribute("r", String(R));
        const isOutput = l === pos.length - 1;
        const [cr, cg, cb] = CLASS_COLORS[i] ?? [200, 200, 200];
        c.setAttribute("fill", isOutput ? `rgb(${cr}, ${cg}, ${cb})` : "#fff");
        c.setAttribute("stroke", "#555");
        this.svg.appendChild(c);
      });
    });
    const label = (text: string, x: number, y: number) => {
      const t = document.createElementNS(SVG_NS, "text");
      t.textContent = text;
      t.setAttribute("x", String(x));
      t.setAttribute("y", String(y));
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("font-size", "11");
      t.setAttribute("fill", "#555");
      this.svg.appendChild(t);
    };
    label("x₁", pos[0][0][0], pos[0][0][1] - 16);
    label("x₂", pos[0][1][0], pos[0][1][1] + 24);
    label("output", pos[pos.length - 1][0][0], marginY - 12);

    // legend for the line colours (they are NOT class colours)
    const legend = (color: string, text: string, x: number) => {
      const l = document.createElementNS(SVG_NS, "line");
      l.setAttribute("x1", String(x));
      l.setAttribute("x2", String(x + 18));
      l.setAttribute("y1", String(H - 10));
      l.setAttribute("y2", String(H - 10));
      l.setAttribute("stroke", color);
      l.setAttribute("stroke-width", "3");
      this.svg.appendChild(l);
      const t = document.createElementNS(SVG_NS, "text");
      t.textContent = text;
      t.setAttribute("x", String(x + 24));
      t.setAttribute("y", String(H - 6));
      t.setAttribute("font-size", "11");
      t.setAttribute("fill", "#555");
      this.svg.appendChild(t);
    };
    legend(POS, "weight > 0", 12);
    legend(NEG, "weight < 0", 112);

    this.update();
  }

  /** Re-colour and re-size every line from the current weights. */
  update() {
    // scale thickness relative to the largest weight in the network
    let maxW = 1e-9;
    for (const { layer, i, j } of this.lines) maxW = Math.max(maxW, Math.abs(layer.W[i][j]));
    for (const { el, layer, i, j } of this.lines) {
      const w = layer.W[i][j];
      const t = Math.abs(w) / maxW;
      el.setAttribute("stroke", w >= 0 ? POS : NEG);
      el.setAttribute("stroke-width", (0.5 + 4.5 * t).toFixed(2));
      el.setAttribute("stroke-opacity", (0.15 + 0.85 * t).toFixed(2));
    }
  }
}
