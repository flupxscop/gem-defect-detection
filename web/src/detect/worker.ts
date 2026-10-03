/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import { ORT_WASM_URL } from "../config";
import type { Prediction } from "../lib/api";
import { fetchModel } from "./modelStore";
import { decodeRtDetr, decodeYolo } from "./postprocess";
import { letterbox, stretch } from "./preprocess";
import type { Backend, WorkerRequest, WorkerResponse } from "./protocol";
import type { ModelSpec, RawDetection, RgbaImage } from "./types";

declare const self: DedicatedWorkerGlobalScope;

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
// Threads need cross-origin isolation (COOP/COEP headers); otherwise ORT falls back to one thread.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

interface Loaded {
  spec: ModelSpec;
  session: ort.InferenceSession;
}

const models = new Map<string, Loaded>();
let queue: Promise<unknown> = Promise.resolve();

const send = (message: WorkerResponse) => self.postMessage(message);

// onnxruntime-web starts its WASM threads from this same script (named "em-pthread");
// those must keep ORT's own message handler.
if (!self.name.startsWith("em-pthread")) {
  self.onmessage = (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    // Handle one message at a time; ORT sessions aren't re-entrant.
    if (message.type === "load") queue = queue.then(() => load(message.baseUrl, message.backends));
    else if (message.type === "predict") queue = queue.then(() => predict(message));
  };
}

async function load(baseUrl: string, backends: Record<string, Backend> = {}) {
  let manifest: Record<string, Omit<ModelSpec, "name">>;
  try {
    if (ORT_WASM_URL) {
      // Hand the runtime over as bytes, fetched and cached like the models.
      ort.env.wasm.wasmBinary = await fetchModel(ORT_WASM_URL, (loaded, total) =>
        send({ type: "progress", model: "runtime", loaded, total }),
      );
    }
    const response = await fetch(`${baseUrl}/manifest.json`, { referrerPolicy: "no-referrer" });
    if (!response.ok) throw new Error(`manifest.json: HTTP ${response.status}`);
    manifest = await response.json();
  } catch (error) {
    send({ type: "load-error", model: "runtime", message: String(error) });
    return;
  }
  // Smallest model first, so the page becomes usable as early as possible.
  const rank = (name: string) => (manifest[name].family === "yolo" ? 0 : 1);
  const order = Object.keys(manifest).sort((a, b) => rank(a) - rank(b));
  for (const name of order) {
    const spec = { name, ...manifest[name] };
    try {
      const bytes = await fetchModel(`${baseUrl}/${spec.file}`, (loaded, total) =>
        send({ type: "progress", model: name, loaded, total }),
      );
      const { session, backend } = await createSession(bytes, backends[name] ?? "webgpu");
      await warmup(session, spec.imgsz);
      models.set(name, { spec, session });
      send({ type: "ready", model: name, backend });
    } catch (error) {
      send({ type: "load-error", model: name, message: String(error) });
    }
  }
}

async function createSession(bytes: ArrayBuffer, preferred: Backend) {
  if (preferred === "webgpu" && "gpu" in navigator) {
    try {
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      return { session, backend: "WebGPU" };
    } catch {
      // Unsupported adapter or op: fall back to WASM below.
    }
  }
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { session, backend: "WASM" };
}

async function warmup(session: ort.InferenceSession, size: number) {
  const input = new ort.Tensor("float32", new Float32Array(3 * size * size), [1, 3, size, size]);
  await session.run({ [session.inputNames[0]]: input });
}

async function predict({ id, image, models: names, conf }: Extract<WorkerRequest, { type: "predict" }>) {
  try {
    const rgba = toRgba(image);
    const predictions: Prediction[] = [];
    for (const name of names) {
      const model = models.get(name);
      if (!model) throw new Error(`Model not loaded: ${name}`);
      predictions.push(await run(model, rgba, conf));
    }
    send({ type: "result", id, predictions });
  } catch (error) {
    send({ type: "predict-error", id, message: String(error) });
  } finally {
    image.close();
  }
}

async function run({ spec, session }: Loaded, image: RgbaImage, conf: number): Promise<Prediction> {
  const started = performance.now();
  const size = spec.imgsz;
  let tensor: Float32Array;
  let restore: (box: RawDetection["box"]) => RawDetection["box"];

  if (spec.family === "yolo") {
    const lb = letterbox(image, size);
    tensor = lb.tensor;
    restore = ([x1, y1, x2, y2]) => [
      clamp((x1 - lb.left) / lb.scale, image.width),
      clamp((y1 - lb.top) / lb.scale, image.height),
      clamp((x2 - lb.left) / lb.scale, image.width),
      clamp((y2 - lb.top) / lb.scale, image.height),
    ];
  } else {
    tensor = stretch(image, size);
    restore = ([x1, y1, x2, y2]) => [x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height];
  }

  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  const output = (await outputs[session.outputNames[0]].getData()) as Float32Array;
  const detections =
    spec.family === "yolo" ? decodeYolo(output, spec.classes.length, conf) : decodeRtDetr(output, conf);

  return {
    model: spec.name as Prediction["model"],
    inference_ms: Math.round((performance.now() - started) * 10) / 10,
    image: { width: image.width, height: image.height },
    detections: detections.map((d) => ({
      class: spec.classes[d.classIndex],
      confidence: Math.round(d.confidence * 1e4) / 1e4,
      box: restore(d.box).map((v) => Math.round(v * 10) / 10) as RawDetection["box"],
    })),
  };
}

function toRgba(bitmap: ImageBitmap): RgbaImage {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return { data, width: bitmap.width, height: bitmap.height };
}

const clamp = (v: number, max: number) => Math.min(Math.max(v, 0), max);
