import { useState } from "react";
import { MODEL_LABELS, ModelName, Selection } from "../lib/api";
import { modelsFor } from "../hooks/usePredictions";
import { SAMPLES } from "../samples";

const SELECTIONS: Selection[] = ["yolo11s", "rtdetr-l", "both"];

interface Props {
  hasFile: boolean;
  busy: boolean;
  selection: Selection;
  available: Set<ModelName> | null;
  conf: number;
  onFile: (file: File | undefined) => void;
  onSample: (url: string) => void;
  onSelection: (selection: Selection) => void;
  onConf: (conf: number) => void;
}

export function UploadStudio(props: Props) {
  const { hasFile, busy, selection, available, conf } = props;
  const [dragging, setDragging] = useState(false);

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    props.onFile(e.target.files?.[0]);
    e.target.value = "";
  };

  return (
    <section id="detect" className="studio" aria-busy={busy}>
      <label
        className={`dropzone ${dragging ? "drag" : ""} ${busy ? "disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) props.onFile(e.dataTransfer.files[0]);
        }}
      >
        <input type="file" accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={pick} />
        <span className="badge" aria-hidden>
          Drop ✱ Here
        </span>
        <span className="micro">Upload</span>
        <strong className="drop-title">
          {hasFile ? (
            <>
              Try <span className="soft">Another</span> Stone
            </>
          ) : (
            <>
              Drop Your <span className="soft">Stone</span>
            </>
          )}
        </strong>
        <span className="micro muted">JPG, PNG or WEBP</span>
        <span className="pill">Browse files</span>
      </label>

      <div className="studio-side">
        <label className={`pill pill-light camera ${busy ? "disabled" : ""}`}>
          <input type="file" accept="image/*" capture="environment" disabled={busy} onChange={pick} />
          Take photo
        </label>

        <fieldset className="segmented" disabled={busy}>
          <legend className="micro">Model</legend>
          <div className="segments">
            {SELECTIONS.map((s) => (
              <label key={s} className={selection === s ? "checked" : ""}>
                <input
                  type="radio"
                  name="model"
                  value={s}
                  checked={selection === s}
                  disabled={available !== null && !modelsFor(s).some((m) => available.has(m))}
                  onChange={() => props.onSelection(s)}
                />
                {s === "both" ? "Both" : MODEL_LABELS[s]}
              </label>
            ))}
          </div>
        </fieldset>

        <label className="slider">
          <span className="slider-head">
            <span className="micro">Confidence ≥</span>
            <span className="serif-num">{conf.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={conf}
            onChange={(e) => props.onConf(Number(e.target.value))}
          />
        </label>

        <div className="samples">
          <span className="micro">Or try a test stone</span>
          <div className="sample-row">
            {SAMPLES.map((url, i) => (
              <button key={url} type="button" disabled={busy} onClick={() => props.onSample(url)}>
                <img src={url} alt={`Test stone ${i + 1}`} loading="lazy" width={64} height={64} />
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
