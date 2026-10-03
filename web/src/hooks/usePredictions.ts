import { useEffect, useRef, useState } from "react";
import { MODEL_NAMES, ModelName, predict, Prediction, Selection } from "../lib/api";

export const modelsFor = (s: Selection): ModelName[] => (s === "both" ? MODEL_NAMES : [s]);

type Results = Partial<Record<ModelName, Prediction>>;
const NO_RESULTS: Results = {};

/**
 * Fetches predictions for the current file, only for selected models not fetched yet.
 * A failed request isn't retried for the same file and selection until the backend reconnects.
 */
export function usePredictions(
  file: File | null,
  selection: Selection,
  available: Set<ModelName>,
  reconnects: number,
  onNetworkError: () => void,
) {
  // Results are stored with the file they belong to, so a new file starts empty without an extra render.
  const [store, setStore] = useState<{ file: File | null; results: Results }>({ file: null, results: NO_RESULTS });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const failed = useRef<{ file: File; selection: Selection; reconnects: number } | null>(null);

  const results = store.file === file ? store.results : NO_RESULTS;

  useEffect(() => {
    setError(null);
  }, [file]);

  useEffect(() => {
    if (!file) return;
    const last = failed.current;
    if (last && last.file === file && last.selection === selection && last.reconnects === reconnects) return;
    const needed = modelsFor(selection).filter((m) => !results[m] && available.has(m));
    if (needed.length === 0) return;

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    predict(file, needed.length > 1 ? "both" : needed[0], controller.signal)
      .then((predictions) =>
        setStore((prev) => ({
          file,
          results: { ...(prev.file === file ? prev.results : {}), ...Object.fromEntries(predictions.map((p) => [p.model, p])) },
        })),
      )
      .catch((e: Error) => {
        if (e.name === "AbortError") return;
        failed.current = { file, selection, reconnects };
        if (e instanceof TypeError) onNetworkError(); // the status banner explains it
        else setError(`Prediction failed: ${e.message}`);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      setLoading(false);
    };
  }, [file, selection, results, available, reconnects, onNetworkError]);

  return { results, loading, error, setError };
}
