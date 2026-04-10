import type { ExplainabilityReportSection, ReportReasoning } from "../types";

interface ExplainabilitySectionProps {
  section?: ExplainabilityReportSection;
  reasoning: ReportReasoning;
  onViewExplainability: () => void;
  onViewCounterfactual: (candidateId: string) => void;
}

function formatConfidence(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

export function ExplainabilitySection(
  props: ExplainabilitySectionProps
): JSX.Element | null {
  const { section, reasoning, onViewExplainability, onViewCounterfactual } = props;

  if (!section) {
    return null;
  }

  return (
    <section className="report-section" aria-labelledby="explainability-title">
      <div className="report-section-head">
        <h3 id="explainability-title">Explainability</h3>
        <p>Evidence cards, reasoning traces, and counterfactuals</p>
      </div>

      <button
        type="button"
        className="report-ghost-action"
        onClick={onViewExplainability}
      >
        Inspect evidence basis
      </button>

      <div className="report-explainability-grid">
        {section.evidence_cards.slice(0, 4).map((card) => (
          <article className="report-explainability-card" key={`${card.candidate_id}-${card.rank}`}>
            <div className="report-explainability-card-head">
              <strong>Comparable #{card.rank}</strong>
              <span>
                Confidence band: {formatConfidence(Number(card.temporal_confidence_band?.p50 ?? NaN))}
              </span>
            </div>
            <pre className="report-explainability-pre">
              {JSON.stringify(card.feature_contributions, null, 2)}
            </pre>
          </article>
        ))}
      </div>

      <div className="report-meaning-block">
        <h4>Reasoning highlights</h4>
        <ul>
          {reasoning.explanation_units.slice(0, 4).map((item) => (
            <li key={item.explanation_id}>{item.statement}</li>
          ))}
        </ul>
      </div>

      <div className="report-counterfactual-list">
        {section.counterfactual_actions.slice(0, 3).map((action) => (
          <article className="report-counterfactual-item" key={`${action.candidate_id}-${action.rank}`}>
            <div className="report-counterfactual-head">
              <strong>Comparable #{action.rank}</strong>
              <button
                type="button"
                className="report-ghost-action"
                onClick={() => onViewCounterfactual(action.candidate_id)}
              >
                View counterfactuals
              </button>
            </div>
            <ul>
              {action.scenarios.slice(0, 3).map((scenario) => (
                <li key={scenario.scenario_id}>
                  {scenario.scenario_id}: {scenario.feasibility}
                  {scenario.reason ? `, ${scenario.reason}` : ""}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>

      <p className="report-section-note">{section.disclaimer}</p>
    </section>
  );
}
