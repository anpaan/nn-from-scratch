import { defineConfig } from "vite";

// For GitHub Pages the site lives at https://<user>.github.io/nn-from-scratch/
// so assets must be served from that sub-path. Change it if you rename the repo.
export default defineConfig({
  base: "/nn-from-scratch/",
});
