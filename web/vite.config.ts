import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation lets onnxruntime-web use multiple WASM threads.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    port: 5173,
    strictPort: true,
    headers: isolation,
    fs: { allow: [".."] }, // results/comparison.json lives outside web/
    proxy: { "/api": "http://localhost:8000" },
  },
  preview: { headers: isolation },
});
