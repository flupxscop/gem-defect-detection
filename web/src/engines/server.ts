import { getHealth, predict } from "../lib/api";
import { Engine, Store } from "./types";

const RETRY_MS = 4000;

/** Calls the FastAPI backend. Polls /api/health while it's unreachable. */
export function createServerEngine(): Engine {
  const store = new Store();
  let timer: number | undefined;

  const refresh = async () => {
    window.clearTimeout(timer);
    try {
      const health = await getHealth();
      store.set({ ready: health.models, offline: false, backend: "Server" });
    } catch {
      if (!store.get().offline || store.get().ready.length) store.set({ ready: [], offline: true });
      timer = window.setTimeout(refresh, RETRY_MS);
    }
  };
  refresh();

  return {
    getStatus: store.get,
    subscribe: store.subscribe,
    predict: (file, models, signal) => predict(file, models.length > 1 ? "both" : models[0], signal),
    retry: refresh,
  };
}
