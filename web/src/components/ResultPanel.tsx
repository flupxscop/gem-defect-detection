import { useEffect, useRef } from "react";
import { Detection, MODEL_LABELS, ModelName, Prediction } from "../lib/api";

// Same colours as results/comparison.png (see src/compare.py).
export const MODEL_COLORS: Record<ModelName, string> = { yolo11s: "#2a78d6", "rtdetr-l": "#eb6834" };

interface Props {
  imageUrl: string;
  prediction: Prediction;
  detections: Detection[];
  highlighted: number | null;
  onHighlight: (index: number | null) => void;
}

/** Image with boxes drawn on an overlay canvas, plus the detection list. */
export function ResultPanel({ imageUrl, prediction, detections, highlighted, onHighlight }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const color = MODEL_COLORS[prediction.model];

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      const width = img.clientWidth;
      const height = img.clientHeight;
      if (!width || !height) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // Boxes come in original-image pixels; scale them to the displayed size.
      const sx = width / prediction.image.width;
      const sy = height / prediction.image.height;
      const font = 12;
      ctx.font = `700 ${font}px Manrope, system-ui, sans-serif`;
      ctx.textBaseline = "top";

      detections.forEach((d, i) => {
        const active = highlighted === i;
        const dimmed = highlighted !== null && !active;
        const [x1, y1, x2, y2] = d.box;
        const x = x1 * sx;
        const y = y1 * sy;
        const w = (x2 - x1) * sx;
        const h = (y2 - y1) * sy;

        ctx.globalAlpha = dimmed ? 0.35 : 1;
        ctx.lineWidth = active ? 4 : 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x, y, w, h);

        const label = `${d.confidence.toFixed(2)}`;
        const tw = ctx.measureText(label).width + 8;
        const ty = y - font - 6 < 0 ? y : y - font - 6;
        ctx.fillStyle = color;
        ctx.fillRect(x - ctx.lineWidth / 2, ty, tw, font + 6);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + 4 - ctx.lineWidth / 2, ty + 3);
      });
      ctx.globalAlpha = 1;
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(img);
    img.addEventListener("load", draw);
    return () => {
      observer.disconnect();
      img.removeEventListener("load", draw);
    };
  }, [imageUrl, prediction, detections, highlighted, color]);

  return (
    <article className="result">
      <header className="result-head">
        <div>
          <span className="micro model-tag">
            <span className="swatch" style={{ background: color }} aria-hidden />
            {prediction.model === "yolo11s" ? "CNN · one-stage" : "Transformer · DETR"}
          </span>
          <h3 className="model-name">{MODEL_LABELS[prediction.model]}</h3>
        </div>
        <dl className="stats">
          <div>
            <dt className="micro">Inclusions</dt>
            <dd className="serif-num">{detections.length}</dd>
          </div>
          <div>
            <dt className="micro">Inference</dt>
            <dd className="serif-num">
              {prediction.inference_ms.toFixed(0)}
              <small>ms</small>
            </dd>
          </div>
        </dl>
      </header>

      <div className="stage">
        <img ref={imgRef} src={imageUrl} alt={`Uploaded gemstone with ${MODEL_LABELS[prediction.model]} detections`} />
        <canvas ref={canvasRef} aria-hidden />
      </div>

      {detections.length === 0 ? (
        <p className="micro muted empty">No inclusions above this confidence.</p>
      ) : (
        <ol className="detections">
          {detections.map((d, i) => (
            <li key={i}>
              <button
                type="button"
                className={highlighted === i ? "active" : ""}
                onClick={() => onHighlight(highlighted === i ? null : i)}
                aria-pressed={highlighted === i}
              >
                <span className="det-index">{String(i + 1).padStart(2, "0")}</span>
                <span className="det-class">{d.class}</span>
                <span className="det-conf">{(d.confidence * 100).toFixed(0)}%</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </article>
  );
}
