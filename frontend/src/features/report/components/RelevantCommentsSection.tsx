import type { RelevantCommentsData } from "../types";

interface RelevantCommentsSectionProps {
  section: RelevantCommentsData;
}

function getPolarityStyle(polarity: string): { bg: string; color: string; icon: string } {
  switch (polarity.toLowerCase()) {
    case "positive":
      return { bg: "rgba(34,197,94,0.16)", color: "#86efac", icon: "👍" };
    case "negative":
    case "negativo":
      return { bg: "rgba(239,68,68,0.16)", color: "#fca5a5", icon: "👎" };
    case "question":
    case "pregunta":
      return { bg: "rgba(96,165,250,0.16)", color: "#93bbfd", icon: "❓" };
    default:
      return { bg: "rgba(148,163,184,0.12)", color: "#94a3b8", icon: "💬" };
  }
}

export function RelevantCommentsSection(
  props: RelevantCommentsSectionProps
): JSX.Element {
  const { section } = props;

  return (
    <section className="report-section" aria-labelledby="comments-title">
      <div className="report-section-head">
        <h3 id="comments-title">Relevant Comments</h3>
        <p>Audience sentiment from comparable videos</p>
      </div>

      <div className="comments-grid">
        {section.items.map((comment) => {
          const polarityStyle = getPolarityStyle(comment.polarity);

          return (
            <article className="comment-card" key={comment.id}>
              <div className="comment-card-body">
                <span className="comment-polarity-icon">{polarityStyle.icon}</span>
                <div>
                  <p className="comment-text">{comment.text}</p>
                  <p className="comment-relevance">{comment.relevance_note}</p>
                </div>
              </div>
              <div className="comment-card-footer">
                <span className="comment-topic-badge">{comment.topic}</span>
                <span
                  className="comment-polarity-badge"
                  style={{ background: polarityStyle.bg, color: polarityStyle.color }}
                >
                  {comment.polarity}
                </span>
              </div>
            </article>
          );
        })}
      </div>

      <p className="report-section-note">{section.disclaimer}</p>
    </section>
  );
}
