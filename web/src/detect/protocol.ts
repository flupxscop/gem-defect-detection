import type { Prediction } from "../lib/api";

export type Backend = "webgpu" | "wasm";

export type WorkerRequest =
  | { type: "load"; baseUrl: string; backends?: Record<string, Backend> }
  | { type: "predict"; id: number; image: ImageBitmap; models: string[]; conf: number };

export type WorkerResponse =
  | { type: "progress"; model: string; loaded: number; total: number }
  | { type: "ready"; model: string; backend: string }
  | { type: "load-error"; model: string; message: string }
  | { type: "result"; id: number; predictions: Prediction[] }
  | { type: "predict-error"; id: number; message: string };
