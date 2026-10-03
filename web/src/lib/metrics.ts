import rows from "../../../results/comparison.json";
import { MODEL_LABELS, ModelName } from "./api";

export interface ModelMetrics {
  name: ModelName;
  label: string;
  architecture: string;
  split: string;
  recall: number;
  mAP50: number;
  mAP5095: number;
  msPerImage: number;
}

/** Test-split results from src/compare.py, bundled at build time. */
export const METRICS: ModelMetrics[] = rows.map((r) => ({
  name: r.model as ModelName,
  label: MODEL_LABELS[r.model as ModelName] ?? r.model,
  architecture: r.architecture,
  split: r.split,
  recall: r.recall,
  mAP50: r.mAP50,
  mAP5095: r["mAP50-95"],
  msPerImage: r.ms_per_image,
}));
