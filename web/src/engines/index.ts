import { useSyncExternalStore } from "react";
import { INFERENCE_MODE } from "../config";
import { createBrowserEngine } from "./browser";
import { createServerEngine } from "./server";
import type { Engine } from "./types";

export const engine: Engine = INFERENCE_MODE === "server" ? createServerEngine() : createBrowserEngine();

export function useEngineStatus() {
  return useSyncExternalStore(engine.subscribe, engine.getStatus);
}

export type { EngineStatus } from "./types";
