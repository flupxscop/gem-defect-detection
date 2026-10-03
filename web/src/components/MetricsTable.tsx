import { ModelSummary } from "../lib/api";
import { MODEL_COLORS } from "./ResultPanel";

const RECALL_TIE = 0.02;
const fmt = (v: number | string | undefined, digits: number) => (typeof v === "number" ? v.toFixed(digits) : "—");
const num = (m: ModelSummary, key: string) => (typeof m.metrics?.[key] === "number" ? (m.metrics[key] as number) : null);

/** Same rule as src/compare.py: highest recall wins; within 0.02 recall, the faster model wins. */
function recommended(models: ModelSummary[]): string | null {
  const rated = models.filter((m) => num(m, "recall") !== null && num(m, "ms_per_image") !== null);
  if (rated.length === 0) return null;
  const best = Math.max(...rated.map((m) => num(m, "recall")!));
  const contenders = rated.filter((m) => best - num(m, "recall")! <= RECALL_TIE);
  return contenders.reduce((a, b) => (num(b, "ms_per_image")! < num(a, "ms_per_image")! ? b : a)).name;
}

/** Test-split metrics from results/comparison.csv, served by GET /api/models. */
export function MetricsTable({ models }: { models: ModelSummary[] }) {
  const hasMetrics = models.some((m) => m.metrics);
  const split = models.find((m) => m.metrics)?.metrics?.split;
  const pick = recommended(models);

  return (
    <section className="band" id="compare">
      <div className="band-inner">
        <div className="section-head">
          <h2 className="display-sm">
            Model <span className="soft">Comparison</span>
          </h2>
          <p className="micro">
            {split ? `Evaluated on the ${split} split. ` : ""}Recall comes first in QC: a missed inclusion costs more than
            a false alarm.
          </p>
        </div>

        {!hasMetrics ? (
          <p className="micro">
            No metrics yet. Run <code>python src/compare.py</code> to create results/comparison.csv.
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col" className="micro">Model</th>
                  <th scope="col" className="micro num">Recall</th>
                  <th scope="col" className="micro num">mAP50</th>
                  <th scope="col" className="micro num wide">mAP50-95</th>
                  <th scope="col" className="micro num">ms / image</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name}>
                    <th scope="row">
                      <span className="row-model">
                        <span className="swatch" style={{ background: MODEL_COLORS[m.name] }} aria-hidden />
                        {m.label}
                      </span>
                      <span className="micro arch">{m.architecture}</span>
                      {pick === m.name && <span className="pick micro">✱ Recommended</span>}
                    </th>
                    <td className="num serif-num">{fmt(m.metrics?.recall, 3)}</td>
                    <td className="num serif-num">{fmt(m.metrics?.mAP50, 3)}</td>
                    <td className="num serif-num wide">{fmt(m.metrics?.["mAP50-95"], 3)}</td>
                    <td className="num serif-num">{fmt(m.metrics?.ms_per_image, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
