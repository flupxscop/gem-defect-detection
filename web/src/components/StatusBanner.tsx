import { INFERENCE_MODE } from "../config";
import type { EngineStatus } from "../engines/types";
import { MODEL_LABELS, MODEL_NAMES } from "../lib/api";

const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(0);

export function StatusBanner({ status, onRetry }: { status: EngineStatus; onRetry: () => void }) {
  if (status.offline) {
    return (
      <div className="alert error" role="alert">
        <strong>Can't reach the detection server.</strong>{" "}
        {import.meta.env.DEV && (
          <>
            Start it from the project root with <code>uvicorn app.main:app --port 8000</code>.{" "}
          </>
        )}
        Retrying automatically, or{" "}
        <button type="button" className="link" onClick={onRetry}>
          retry now
        </button>
        .
      </div>
    );
  }

  if (status.errors.length > 0) {
    return (
      <div className="alert error" role="alert">
        {status.errors.join(" ")}
      </div>
    );
  }

  if (status.downloads.length > 0) {
    return (
      <div className="alert info" role="status">
        <span className="spinner" aria-hidden />
        <span>
          {status.downloads.map((d) => (
            <span key={d.model} className="download">
              Loading {d.model === "runtime" ? "inference runtime" : MODEL_LABELS[d.model]} · {mb(d.loaded)}
              {d.total ? ` / ${mb(d.total)}` : ""} MB
            </span>
          ))}
          <span className="micro muted"> Models run in your browser and are cached after the first visit.</span>
        </span>
      </div>
    );
  }

  const missing = MODEL_NAMES.filter((m) => !status.ready.includes(m));
  if (INFERENCE_MODE === "server" && status.backend && missing.length > 0) {
    return (
      <div className="alert warn" role="status">
        <strong>{missing.map((m) => MODEL_LABELS[m]).join(" and ")} not available on the server.</strong>
      </div>
    );
  }
  return null;
}
