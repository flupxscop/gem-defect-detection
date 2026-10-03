import { METRICS, ModelMetrics } from "../lib/metrics";
import { MODEL_COLORS } from "./ResultPanel";

const RECALL_TIE = 0.02;

/** Same rule as src/compare.py: highest recall wins; within 0.02 recall, the faster model wins. */
function recommended(models: ModelMetrics[]): string | null {
  if (models.length === 0) return null;
  const best = Math.max(...models.map((m) => m.recall));
  const contenders = models.filter((m) => best - m.recall <= RECALL_TIE);
  return contenders.reduce((a, b) => (b.msPerImage < a.msPerImage ? b : a)).name;
}

export function MetricsTable() {
  if (METRICS.length === 0) return null;
  const pick = recommended(METRICS);

  return (
    <section className="band" id="compare">
      <div className="band-inner">
        <div className="section-head">
          <h2 className="display-sm">
            Model <span className="soft">Comparison</span>
          </h2>
          <p className="micro">
            Evaluated on the {METRICS[0].split} split. Recall comes first in QC: a missed inclusion costs more than a
            false alarm.
          </p>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col" className="micro">Model</th>
                <th scope="col" className="micro num">Recall</th>
                <th scope="col" className="micro num">mAP50</th>
                <th scope="col" className="micro num wide">mAP50-95</th>
                <th scope="col" className="micro num">GPU ms</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => (
                <tr key={m.name}>
                  <th scope="row">
                    <span className="row-model">
                      <span className="swatch" style={{ background: MODEL_COLORS[m.name] }} aria-hidden />
                      {m.label}
                    </span>
                    <span className="micro arch">{m.architecture}</span>
                    {pick === m.name && <span className="pick micro">✱ Recommended</span>}
                  </th>
                  <td className="num serif-num">{m.recall.toFixed(3)}</td>
                  <td className="num serif-num">{m.mAP50.toFixed(3)}</td>
                  <td className="num serif-num wide">{m.mAP5095.toFixed(3)}</td>
                  <td className="num serif-num">{m.msPerImage.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="micro band-note">GPU ms: Apple M2 during evaluation. Live timings appear on each result.</p>
      </div>
    </section>
  );
}
