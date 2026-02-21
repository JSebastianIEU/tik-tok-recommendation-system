import type { RelevantCommentsData } from "../types";

interface RelevantCommentsSectionProps {
  section: RelevantCommentsData;
}

export function RelevantCommentsSection(
  props: RelevantCommentsSectionProps
): JSX.Element {
  const { section } = props;

  return (
    <section className="report-section" aria-labelledby="comments-title">
      <div className="report-section-head">
        <h3 id="comments-title">Relevant comments</h3>
      </div>

      <div className="comment-list">
        {section.items.map((comment) => (
          <article className="comment-item" key={comment.id}>
            <p>{comment.text}</p>
            <p className="comment-relevance-note">{comment.relevance_note}</p>

            <div className="comment-meta">
              <span className="comment-topic">Topic: {comment.topic}</span>
              <span
                className={`comment-polarity comment-polarity-${comment.polarity.toLowerCase()}`}
              >
                {comment.polarity}
              </span>
            </div>
          </article>
        ))}
      </div>

      <p className="report-section-note">{section.disclaimer}</p>
    </section>
  );
}
