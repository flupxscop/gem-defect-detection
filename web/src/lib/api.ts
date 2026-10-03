// Server mode only. Same origin by default: the backend serves the app, and Vite proxies /api in development.
const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export type ModelName = "yolo11s" | "rtdetr-l";
export type Selection = ModelName | "both";

export const MODEL_NAMES: ModelName[] = ["yolo11s", "rtdetr-l"];
export const MODEL_LABELS: Record<ModelName, string> = { yolo11s: "YOLO11s", "rtdetr-l": "RT-DETR-L" };

export interface Detection {
  class: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface Prediction {
  model: ModelName;
  inference_ms: number;
  image: { width: number; height: number };
  detections: Detection[];
}

export interface Health {
  status: "ok" | "no_models";
  models: ModelName[];
}

// Request every box above this once; the confidence slider then filters locally.
export const FETCH_CONF = 0.05;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json() as Promise<T>;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const { detail } = await res.json();
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // Not JSON; fall through to the status line.
  }
  return `${res.status} ${res.statusText}`;
}

export const getHealth = () => request<Health>("/api/health");

export function predict(file: File, model: Selection, signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("conf", String(FETCH_CONF));
  return request<Prediction[]>("/api/predict", { method: "POST", body: form, signal });
}
