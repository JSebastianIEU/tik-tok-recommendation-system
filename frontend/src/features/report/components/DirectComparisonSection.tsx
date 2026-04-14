import type { DirectComparisonData } from "../types";

interface DirectComparisonSectionProps {
  comparison: DirectComparisonData;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function DirectComparisonSection(
  props: DirectComparisonSectionProps
): JSX.Element {
  const { comparison } = props;

  return (
    <section className="report-section" aria-labelledby="direct-comparison-title">
      <div className="report-section-head">
        <h3 id="direct-comparison-title">Your Video vs. Comparables</h3>
        <p>Side-by-side performance benchmarks</p>
      </div>

      <div className="dc-list">
        {comparison.rows.map((row) => {
          const yourPct = clampPercent(row.your_value_pct);
          const avgPct = clampPercent(row.comparable_value_pct);
          const diff = yourPct - avgPct;

          return (
            <article className="dc-row" key={row.id}>
              <div className="dc-row-label">{row.label}</div>

              <div className="dc-bars">
                <div className="dc-bar-group">
                  <div className="dc-bar-header">
                    <span className="dc-bar-tag dc-bar-tag-you">You</span>
                    <span className="dc-bar-val">{row.your_value_label}</span>
                  </div>
                  <div className="dc-track">
                    <div
                      className="dc-fill dc-fill-you"
                      style={{ width: `${yourPct}%` }}
                    />
                  </div>
                </div>

                <div className="dc-bar-group">
                  <div className="dc-bar-header">
                    <span className="dc-bar-tag dc-bar-tag-avg">Avg</span>
                    <span className="dc-bar-val">{row.comparable_value_label}</span>
                  </div>
                  <div className="dc-track">
                    <div
                      className="dc-fill dc-fill-avg"
                      style={{ width: `${avgPct}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className={`dc-diff ${diff >= 0 ? "dc-diff-up" : "dc-diff-down"}`}>
                {diff >= 0 ? "+" : ""}{Math.round(diff)}%
              </div>
            </article>
          );
        })}
      </div>

      <p className="report-section-note">{comparison.note}</p>
    </section>
  );
}
