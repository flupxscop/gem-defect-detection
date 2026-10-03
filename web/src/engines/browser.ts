import { MODEL_BASE_URL } from "../config";
import type { WorkerRequest, WorkerResponse } from "../detect/protocol";
import { FETCH_CONF, ModelName, Prediction } from "../lib/api";
import { Engine, Store } from "./types";

/** Runs the ONNX models in a Web Worker with onnxruntime-web (WebGPU, falling back to WASM). */
export function createBrowserEngine(): Engine {
  const store = new Store();
  const worker = new Worker(new URL("../detect/worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, { resolve: (p: Prediction[]) => void; reject: (e: Error) => void }>();
  let nextId = 0;

  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    const status = store.get();
    switch (data.type) {
      case "progress": {
        const others = status.downloads.filter((d) => d.model !== data.model);
        store.set({ downloads: [...others, { model: data.model as ModelName, loaded: data.loaded, total: data.total }] });
        break;
      }
      case "ready":
        store.set({
          ready: [...status.ready, data.model as ModelName],
          downloads: status.downloads.filter((d) => d.model !== data.model),
          backend: data.backend,
        });
        break;
      case "load-error":
        store.set({
          downloads: status.downloads.filter((d) => d.model !== data.model),
          errors: [...status.errors, `Couldn't load ${data.model}: ${data.message}`],
        });
        break;
      case "result":
        pending.get(data.id)?.resolve(data.predictions);
        pending.delete(data.id);
        break;
      case "predict-error":
        pending.get(data.id)?.reject(new Error(data.message));
        pending.delete(data.id);
        break;
    }
  };

  const post = (message: WorkerRequest, transfer: Transferable[] = []) => worker.postMessage(message, transfer);
  post({ type: "load", baseUrl: MODEL_BASE_URL });

  return {
    getStatus: store.get,
    subscribe: store.subscribe,
    async predict(file, models, signal) {
      const image = await createImageBitmap(file, { imageOrientation: "from-image" });
      const id = nextId++;
      return new Promise<Prediction[]>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        // A running inference can't be cancelled; an aborted request just drops its result.
        signal?.addEventListener("abort", () => {
          pending.delete(id);
          reject(new DOMException("Aborted", "AbortError"));
        });
        post({ type: "predict", id, image, models, conf: FETCH_CONF }, [image]);
      });
    },
    retry() {},
  };
}
