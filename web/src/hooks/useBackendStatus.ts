import { useCallback, useEffect, useState } from "react";
import { getHealth, getModels, Health, ModelSummary } from "../lib/api";

const RETRY_MS = 4000;

export interface BackendStatus {
  health: Health | null;
  models: ModelSummary[];
  offline: boolean;
  /** Increments each time the backend comes back after being offline. */
  reconnects: number;
  refresh: () => void;
}

export function useBackendStatus(): BackendStatus {
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [offline, setOffline] = useState(false);
  const [reconnects, setReconnects] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([getHealth(), getModels()]);
      setHealth(h);
      setModels(m);
      setOffline((wasOffline) => {
        if (wasOffline) setReconnects((n) => n + 1);
        return false;
      });
    } catch {
      setOffline(true);
      setHealth(null);
      setModels([]);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Keep polling while offline, e.g. while a sleeping host wakes up.
  useEffect(() => {
    if (!offline) return;
    const id = window.setInterval(refresh, RETRY_MS);
    return () => window.clearInterval(id);
  }, [offline, refresh]);

  return { health, models, offline, reconnects, refresh };
}
