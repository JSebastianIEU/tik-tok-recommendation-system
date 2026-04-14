import type { RecommendationsData } from "../types";

interface RecommendationsSectionProps {
  section: RecommendationsData;
  onFeedback: (
    recommendationId: string,
    label: "useful" | "not_useful" | "saved"
  ) => void;
  feedbackState: Record<string, "useful" | "not_useful" | "saved" | undefined>;
}

function getPriorityStyle(priority: string): { bg: string; color: string } {
  switch (priority) {
    case "High": return { bg: "rgba(239,68,68,0.18)", color: "#fca5a5" };
    case "Medium": return { bg: "rgba(251,191,36,0.18)", color: "#fde68a" };
    case "Low": return { bg: "rgba(34,197,94,0.18)", color: "#86efac" };
    default: return { bg: "rgba(148,163,184,0.15)", color: "#cbd5e1" };
  }
}

function getEffortStyle(effort: string): { bg: string; color: string } {
  switch (effort) {
    case "Low": return { bg: "rgba(34,197,94,0.15)", color: "#86efac" };
    case "Medium": return { bg: "rgba(251,191,36,0.15)", color: "#fde68a" };
    case "High": return { bg: "rgba(239,68,68,0.15)", color: "#fca5a5" };
    default: return { bg: "rgba(148,163,184,0.12)", color: "#cbd5e1" };
  }
}

function getEffectIcon(area: string): string {
  switch (area) {
    case "hook": return "🎣";
    case "clarity": return "🔍";
    case "cta": return "📢";
    case "pacing": return "⏱️";
    case "format": return "🎬";
    case "audience_alignment": return "🎯";
    case "topic_alignment": return "📌";
    default: return "💡";
  }
}

export function RecommendationsSection(
  props: RecommendationsSectionProps
): JSX.Element {
  const { section } = props;

  return (
    <section className="report-section" aria-labelledby="recommendations-title">
      <div className="report-section-head">
        <h3 id="recommendations-title">Recommendations</h3>
        <p>Actionable steps to improve performance</p>
      </div>

      <div className="recs-list">
        {section.items.map((rec) => {
          const priorityStyle = getPriorityStyle(rec.priority);
          const effortStyle = getEffortStyle(rec.effort);
          const fbState = props.feedbackState[rec.id];

          return (
            <article className="rec-card" key={rec.id}>
              <div className="rec-card-top">
                <span className="rec-effect-icon">{getEffectIcon(rec.effect_area)}</span>
                <div className="rec-card-title-block">
                  <h4>{rec.title}</h4>
                  <span className="rec-effect-area">{rec.effect_area.replace(/_/g, " ")}</span>
                </div>
                <div className="rec-badges">
                  <span
                    className="rec-badge"
                    style={{ background: priorityStyle.bg, color: priorityStyle.color }}
                  >
                    {rec.priority}
                  </span>
                  <span
                    className="rec-badge"
                    style={{ background: effortStyle.bg, color: effortStyle.color }}
                  >
                    {rec.effort} effort
                  </span>
                </div>
              </div>

              <div className="rec-body">
                <p className="rec-evidence">{rec.evidence}</p>
                <p className="rec-rationale">{rec.rationale}</p>

                {rec.caveats.length > 0 && (
                  <div className="rec-caveats">
                    <span className="rec-caveats-label">Caveats:</span>
                    {rec.caveats.map((c, i) => (
                      <span className="rec-caveat-chip" key={i}>{c}</span>
                    ))}
                  </div>
                )}

                <div className="rec-confidence">
                  <span className="rec-confidence-dot" />
                  {rec.confidence_label}
                </div>
              </div>

              <div className="rec-actions">
                <button
                  type="button"
                  className={`rec-action-btn ${fbState === "useful" ? "rec-action-active-good" : ""}`}
                  onClick={() => props.onFeedback(rec.id, "useful")}
                >
                  {fbState === "useful" ? "Useful ✓" : "Useful"}
                </button>
                <button
                  type="button"
                  className={`rec-action-btn ${fbState === "not_useful" ? "rec-action-active-bad" : ""}`}
                  onClick={() => props.onFeedback(rec.id, "not_useful")}
                >
                  {fbState === "not_useful" ? "Not useful ✗" : "Not useful"}
                </button>
                <button
                  type="button"
                  className={`rec-action-btn ${fbState === "saved" ? "rec-action-active-good" : ""}`}
                  onClick={() => props.onFeedback(rec.id, "saved")}
                >
                  {fbState === "saved" ? "Saved ✓" : "Save"}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
