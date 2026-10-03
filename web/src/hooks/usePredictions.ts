import { useEffect, useRef, useState } from "react";
import type { Engine, EngineStatus } from "../engines/types";
import { MODEL_NAMES, ModelName, Prediction, Selection } from "../lib/api";

export const modelsFor = (s: Selection): ModelName[] => (s === "both" ? MODEL_NAMES : [s]);

type Results = Partial<Record<ModelName, Prediction>>;
const NO_RESULTS: Results = {};

/**
 * Runs the selected models on the current file, only those without results yet and ready to use.
 * A failed request isn't retried for the same file and selection until the engine status changes.
 */
export function usePredictions(engine: Engine, status: EngineStatus, file: File | null, selection: Selection) {
  // Results are stored with the file they belong to, so a new file starts empty without an extra render.
  const [store, setStore] = useState<{ file: File | null; results: Results }>({ file: null, results: NO_RESULTS });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const failed = useRef<{ file: File; selection: Selection; version: number } | null>(null);

  const results = store.file === file ? store.results : NO_RESULTS;

  useEffect(() => {
    setError(null);
  }, [file]);

  useEffect(() => {
    if (!file) return;
    const last = failed.current;
    if (last && last.file === file && last.selection === selection && last.version === status.version) return;
    const needed = modelsFor(selection).filter((m) => !results[m] && status.ready.includes(m));
    if (needed.length === 0) return;

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    engine
      .predict(file, needed, controller.signal)
      .then((predictions) =>
        setStore((prev) => ({
          file,
          results: {
            ...(prev.file === file ? prev.results : {}),
            ...Object.fromEntries(predictions.map((p) => [p.model, p])),
          },
        })),
      )
      .catch((e: Error) => {
        if (e.name === "AbortError") return;
        failed.current = { file, selection, version: status.version };
        if (e instanceof TypeError) engine.retry(); // network failure: the status banner explains it
        else setError(`Prediction failed: ${e.message}`);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      setLoading(false);
    };
  }, [engine, file, selection, results, status.ready, status.version]);

  return { results, loading, error, setError };
}
