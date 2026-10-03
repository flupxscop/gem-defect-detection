import type { ModelName, Prediction } from "../lib/api";

export interface Download {
  model: ModelName | "runtime";
  loaded: number;
  total: number;
}

export interface EngineStatus {
  ready: ModelName[];
  downloads: Download[];
  /** Server mode: backend unreachable. Browser mode: never set. */
  offline: boolean;
  /** Where inference runs, e.g. "WebGPU", "WASM", "Server". */
  backend: string | null;
  errors: string[];
  /** Bumps whenever availability changes, so failed requests may be retried. */
  version: number;
}

export interface Engine {
  getStatus(): EngineStatus;
  subscribe(listener: () => void): () => void;
  predict(image: File, models: ModelName[], signal?: AbortSignal): Promise<Prediction[]>;
  retry(): void;
}

export const INITIAL_STATUS: EngineStatus = {
  ready: [],
  downloads: [],
  offline: false,
  backend: null,
  errors: [],
  version: 0,
};

export class Store {
  private status = INITIAL_STATUS;
  private listeners = new Set<() => void>();

  get = () => this.status;

  set(update: Partial<EngineStatus>) {
    this.status = { ...this.status, ...update, version: this.status.version + 1 };
    this.listeners.forEach((l) => l());
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
}
