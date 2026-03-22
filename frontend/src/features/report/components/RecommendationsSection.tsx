import type { RecommendationsData } from "../types";

interface RecommendationsSectionProps {
  section: RecommendationsData;
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
          </article>
        ))}
      </div>
    </section>
  );
}
