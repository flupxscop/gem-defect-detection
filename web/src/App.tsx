import { useCallback, useEffect, useMemo, useState } from "react";
import { MetricsTable } from "./components/MetricsTable";
import { ResultPanel } from "./components/ResultPanel";
import { StatusBanner } from "./components/StatusBanner";
import { UploadStudio } from "./components/UploadStudio";
import { INFERENCE_MODE } from "./config";
import { engine, useEngineStatus } from "./engines";
import { modelsFor, usePredictions } from "./hooks/usePredictions";
import { MODEL_NAMES, ModelName, Selection } from "./lib/api";
import { prepareUpload, UploadError } from "./lib/image";

type Highlight = { model: ModelName; index: number } | null;

export default function App() {
  const status = useEngineStatus();
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>("both");
  const [conf, setConf] = useState(0.25);
  const [highlight, setHighlight] = useState<Highlight>(null);
  const [preparing, setPreparing] = useState(false);

  const available = useMemo(() => new Set(status.ready), [status.ready]);
  const { results, loading, error, setError } = usePredictions(engine, status, file, selection);

  useEffect(() => () => void (imageUrl && URL.revokeObjectURL(imageUrl)), [imageUrl]);

  const acceptFile = useCallback(
    async (picked: File | undefined) => {
      if (!picked) return;
      setPreparing(true);
      try {
        const upload = await prepareUpload(picked);
        setHighlight(null);
        setFile(upload);
        setImageUrl(URL.createObjectURL(picked));
      } catch (e) {
        setError(e instanceof UploadError ? e.message : "Couldn't read that image.");
      } finally {
        setPreparing(false);
      }
    },
    [setError],
  );

  const loadSample = async (url: string) => {
    try {
      const blob = await (await fetch(url)).blob();
      await acceptFile(new File([blob], url.split("/").pop()!, { type: blob.type }));
    } catch {
      setError("Couldn't load that sample.");
    }
  };

  const busy = loading || preparing;
  const shown = modelsFor(selection).filter((m) => results[m]);
  const statusText = status.offline
    ? "Offline"
    : status.ready.length === 0
      ? "Loading models"
      : `${status.ready.length}/${MODEL_NAMES.length} models · ${status.backend}`;

  return (
    <div className="page">
      <nav className="nav">
        <a className="logo" href="#top">
          <span aria-hidden>✱</span> GEMSCAN
        </a>
        <div className="nav-links">
          <a href="#detect">Detect</a>
          <a href="#compare">Compare</a>
        </div>
        <span className={`status-pill ${status.offline ? "off" : status.ready.length ? "" : "pending"}`}>
          <span className="dot" aria-hidden />
          {statusText}
        </span>
      </nav>

      <header className="hero" id="top">
        <p className="micro hero-eyebrow">
          Inclusion detection
          <br />
          for jewelry QC
        </p>
        <h1 className="display">
          <span className="soft">Find</span> Every
          <br />– Inclusion
          <br />
          <span className="soft">In A Glance</span>
        </h1>
        <div className="hero-side">
          <p className="micro">
            Two detectors, one stone. Compare a CNN (YOLO11s) and a Transformer (RT-DETR-L) side by side on your own
            gemstone photos.
          </p>
          <a className="more" href="#detect">
            Start detecting <span aria-hidden>→</span>
          </a>
        </div>
      </header>

      <StatusBanner status={status} onRetry={engine.retry} />

      <UploadStudio
        hasFile={file !== null}
        busy={busy}
        selection={selection}
        available={status.ready.length || status.offline ? available : null}
        conf={conf}
        onFile={acceptFile}
        onSample={loadSample}
        onSelection={(s) => {
          setSelection(s);
          setHighlight(null);
        }}
        onConf={(c) => {
          setConf(c);
          setHighlight(null);
        }}
      />

      {error && (
        <div className="alert error" role="alert">
          {error}
        </div>
      )}

      {busy && (
        <div className="alert info" role="status">
          <span className="spinner" aria-hidden /> {preparing ? "Preparing image…" : "Detecting inclusions…"}
        </div>
      )}

      {imageUrl && shown.length > 0 && (
        <section className="results-section">
          <div className="section-head">
            <h2 className="display-sm">
              The <span className="soft">Result</span>
            </h2>
            <p className="micro">Tap a detection to highlight its box. Move the slider to filter by confidence.</p>
          </div>
          <div className={`results ${shown.length > 1 ? "two" : ""}`}>
            {shown.map((m) => {
              const prediction = results[m]!;
              return (
                <ResultPanel
                  key={m}
                  imageUrl={imageUrl}
                  prediction={prediction}
                  detections={prediction.detections.filter((d) => d.confidence >= conf)}
                  highlighted={highlight?.model === m ? highlight.index : null}
                  onHighlight={(index) => setHighlight(index === null ? null : { model: m, index })}
                />
              );
            })}
          </div>
        </section>
      )}

      <MetricsTable />

      <footer className="footer">
        <span className="logo">
          <span aria-hidden>✱</span> GEMSCAN
        </span>
        <p className="micro">
          {INFERENCE_MODE === "browser"
            ? "Detection runs on your device. Images never leave your browser."
            : "Uploaded images are processed in memory and never stored."}
        </p>
      </footer>
    </div>
  );
}
