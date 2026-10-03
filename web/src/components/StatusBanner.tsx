import { Health, MODEL_LABELS, MODEL_NAMES } from "../lib/api";

interface Props {
  health: Health | null;
  offline: boolean;
  onRetry: () => void;
}

export function StatusBanner({ health, offline, onRetry }: Props) {
  if (offline) {
    return (
      <div className="alert error" role="alert">
        <strong>Can't reach the detection server.</strong>{" "}
        {import.meta.env.DEV ? (
          <>
            Start it from the project root with <code>uvicorn app.main:app --port 8000</code>.
          </>
        ) : (
          <>It may be waking up after a period of inactivity, which takes about a minute.</>
        )}{" "}
        Retrying automatically, or{" "}
        <button type="button" className="link" onClick={onRetry}>
          retry now
        </button>
        .
      </div>
    );
  }

  const missing = health ? MODEL_NAMES.filter((m) => !health.models.includes(m)) : [];
  if (missing.length === 0) return null;
  return (
    <div className="alert warn" role="status">
      <strong>{missing.map((m) => MODEL_LABELS[m]).join(" and ")} not available.</strong>{" "}
      {import.meta.env.DEV && (
        <>
          Export it with <code>python src/export.py</code> and restart the server.
        </>
      )}
    </div>
  );
}
