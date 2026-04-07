import type { RecommendationsData } from "../types";

interface RecommendationsSectionProps {
  section: RecommendationsData;
  onFeedback: (
    recommendationId: string,
    label: "useful" | "not_useful" | "saved"
  ) => void;
  feedbackState: Record<string, "useful" | "not_useful" | "saved" | undefined>;
}

export function RecommendationsSection(
  props: RecommendationsSectionProps
): JSX.Element {
  const { section } = props;

  return (
    <section className="report-section" aria-labelledby="recommendations-title">
      <div className="report-section-head">
        <h3 id="recommendations-title">Recommendations</h3>
      </div>

      <div className="recommendations-list">
        {section.items.map((recommendation) => (
          <article className="recommendation-item" key={recommendation.id}>
            <div className="recommendation-item-head">
              <h4>{recommendation.title}</h4>
              <div className="recommendation-badges">
                <span className="recommendation-badge">
                  Priority: {recommendation.priority}
                </span>
                <span className="recommendation-badge">
                  Effort: {recommendation.effort}
                </span>
              </div>
            </div>

            <p className="recommendation-evidence">{recommendation.evidence}</p>
            <p className="recommendation-evidence">{recommendation.rationale}</p>
            <p className="recommendation-evidence">
              {recommendation.confidence_label} · Effect area: {recommendation.effect_area}
            </p>
            {recommendation.caveats.length > 0 ? (
              <p className="recommendation-evidence">
                Caveats: {recommendation.caveats.join(", ")}
              </p>
            ) : null}
            <div className="recommendation-feedback-row">
              <button
                type="button"
                className="report-ghost-action"
                onClick={() => props.onFeedback(recommendation.id, "useful")}
              >
                {props.feedbackState[recommendation.id] === "useful"
                  ? "Marked useful"
                  : "Useful"}
              </button>
              <button
                type="button"
                className="report-ghost-action"
                onClick={() => props.onFeedback(recommendation.id, "not_useful")}
              >
                {props.feedbackState[recommendation.id] === "not_useful"
                  ? "Marked not useful"
                  : "Not useful"}
              </button>
              <button
                type="button"
                className="report-ghost-action"
                onClick={() => props.onFeedback(recommendation.id, "saved")}
              >
                {props.feedbackState[recommendation.id] === "saved"
                  ? "Saved"
                  : "Save recommendation"}
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
