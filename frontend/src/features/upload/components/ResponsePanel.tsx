import type { VideoAnalysisResult } from "../../../services/contracts/models";

interface ResponsePanelProps {
  analysis: VideoAnalysisResult | null;
}

interface MetricRowProps {
  label: string;
  value: number;
}

function MetricRow(props: MetricRowProps): JSX.Element {
  const { label, value } = props;
  return (
    <li className="metric-row">
      <div className="metric-label-row">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="metric-track" aria-hidden="true">
        <div className="metric-fill" style={{ width: `${value}%` }} />
      </div>
    </li>
  );
}

export function ResponsePanel(props: ResponsePanelProps): JSX.Element {
  const { analysis } = props;

  if (!analysis) {
    return (
      <section className="glass-card response-panel placeholder-panel">
        <h3 className="panel-title">response</h3>
        <p className="placeholder-text">
          Your video analysis will appear here after upload.
        </p>
      </section>
    );
  }

  return (
    <section className="glass-card response-panel">
      <h3 className="panel-title">response</h3>

      <div className="panel-block">
        <h4 className="panel-subtitle">summary</h4>
        <p className="summary-text">{analysis.summary}</p>
      </div>

      <div className="panel-block">
        <h4 className="panel-subtitle">key topics</h4>
        <div className="topic-pill-wrap">
          {analysis.keyTopics.map((topic) => (
            <span className="topic-pill" key={topic}>
              {topic}
            </span>
          ))}
        </div>
      </div>

      <div className="panel-block">
        <h4 className="panel-subtitle">suggested edits</h4>
        <ul className="suggested-list">
          {analysis.suggestedEdits.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <div className="panel-block">
        <h4 className="panel-subtitle">metrics</h4>
        <ul className="metrics-list">
          <MetricRow label="retention" value={analysis.metrics.retention} />
          <MetricRow label="hook" value={analysis.metrics.hookStrength} />
          <MetricRow label="clarity" value={analysis.metrics.clarity} />
        </ul>
      </div>
    </section>
  );
}
