import type { ExecutiveSummaryData } from "../types";

interface ExecutiveSummarySectionProps {
  summary: ExecutiveSummaryData;
}

export function ExecutiveSummarySection(
  props: ExecutiveSummarySectionProps
): JSX.Element {
  const { summary } = props;

  return (
    <section className="report-section" aria-labelledby="report-summary-title">
      <div className="report-section-head">
        <h3 id="report-summary-title">Executive summary</h3>
      </div>

      <div className="report-summary-grid">
        {summary.metrics.map((metric) => (
          <article className="report-summary-card" key={metric.id}>
            <p className="report-summary-label">{metric.label}</p>
            <p className="report-summary-value">{metric.value}</p>
          </article>
        ))}
      </div>

      <div className="report-keywords-block">
        <h4>Extracted keywords</h4>
        <div className="report-keywords-list">
          {summary.extracted_keywords.map((keyword) => (
            <span className="report-keyword-chip" key={keyword}>
              {keyword}
            </span>
          ))}
        </div>
      </div>

      <div className="report-meaning-block">
        <h4>What this means for your video</h4>
        <ul>
          {summary.meaning_points.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
