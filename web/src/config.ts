export type InferenceMode = "browser" | "server";

/** "browser" runs the models client-side (static hosting); "server" calls the FastAPI backend. */
export const INFERENCE_MODE: InferenceMode = import.meta.env.VITE_INFERENCE === "server" ? "server" : "browser";
// Set VITE_INFERENCE=server at build time for the Docker image, where FastAPI serves the app and the API.

const MODEL_REPO = import.meta.env.VITE_MODEL_REPO ?? "ChantaroNtw/gemscan-models";
const MODEL_REVISION = import.meta.env.VITE_MODEL_REVISION ?? "d2fbb1f5c061f0bd458de3fc1d0e9dac9aa456af";
export const MODEL_BASE_URL = `https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}`;

// onnxruntime-web's WebGPU build needs a 25.5 MiB WASM file. It is bundled by default; hosts with a smaller
// per-file limit (Cloudflare: 25 MiB) load the identical file from npm via jsDelivr instead.
export const ORT_WASM_URL: string | undefined = import.meta.env.VITE_ORT_WASM_URL || undefined;
