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
        <h3 id="direct-comparison-title">Direct comparison</h3>
      </div>

      <div className="comparison-list">
        {comparison.rows.map((row) => (
          <article className="comparison-row" key={row.id}>
            <div className="comparison-row-head">
              <span>{row.label}</span>
              <span>
                Your video {row.your_value_label} | Comparable average {" "}
                {row.comparable_value_label}
              </span>
            </div>

            <div className="comparison-track-group">
              <div className="comparison-track comparison-track-you">
                <div
                  className="comparison-fill comparison-fill-you"
                  style={{ width: `${clampPercent(row.your_value_pct)}%` }}
                />
              </div>

              <div className="comparison-track comparison-track-avg">
                <div
                  className="comparison-fill comparison-fill-avg"
                  style={{ width: `${clampPercent(row.comparable_value_pct)}%` }}
                />
              </div>
            </div>
          </article>
        ))}
      </div>

      <p className="report-section-note">{comparison.note}</p>
    </section>
  );
}
