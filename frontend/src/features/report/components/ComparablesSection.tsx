import type { ComparableItem } from "../types";
import { ComparableThumbnailImage } from "./ComparableThumbnailImage";

interface ComparablesSectionProps {
  items: ComparableItem[];
  selectedComparableId: string | null;
  onSelectComparable: (id: string) => void;
  onOpenVideo: (item: ComparableItem) => void;
  onMarkRelevant: (item: ComparableItem, label: "relevant" | "not_relevant") => void;
  onSaveComparable: (item: ComparableItem) => void;
  feedbackState: Record<string, "relevant" | "not_relevant" | "saved" | undefined>;
}

function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

function formatSimilarity(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function ComparablesSection(props: ComparablesSectionProps): JSX.Element {
  const { items, selectedComparableId, onSelectComparable } = props;

  return (
    <section
      className="report-section report-comparables-section"
      aria-labelledby="comparables-title"
    >
      <div className="report-section-head">
        <h3 id="comparables-title">Comparable videos</h3>
        <p>Retrieved from local-dataset similarity signals</p>
      </div>

      <div className="comparable-list" role="list">
        {items.map((item, itemIndex) => {
          const isActive = selectedComparableId === item.id;
          const hasVideoUrl = item.video_url.trim().length > 0;

          return (
            <article
              className={`comparable-item ${isActive ? "comparable-item-active" : ""}`}
              key={`${item.id}-${itemIndex}`}
              role="listitem"
            >
              {hasVideoUrl ? (
                <a
                  href={item.video_url}
                  target="_blank"
                  rel="noreferrer"
                  className="comparable-thumb-link"
                  aria-label="Open comparable video on TikTok"
                  onClick={() => props.onOpenVideo(item)}
                >
                  <ComparableThumbnailImage
                    className="comparable-thumb-image"
                    thumbnailUrl={item.thumbnail_url}
                    videoUrl={item.video_url}
                    alt="Comparable thumbnail"
                    fallbackClassName="comparable-thumb comparable-thumb-fallback"
                  />
                  <span className="comparable-thumb-overlay">Open</span>
                </a>
              ) : (
                <span className="comparable-thumb comparable-thumb-fallback" aria-hidden="true" />
              )}

              <div className="comparable-main-shell">
                <button
                  type="button"
                  className="comparable-main-button"
                  onClick={() => onSelectComparable(item.id)}
                  aria-label="View comparable details"
                >
                  <span className="comparable-main">
                    <span className="comparable-main-top">
                      <span className="comparable-score">
                        Similarity: {formatSimilarity(item.similarity)}
                      </span>
                      <span className="comparable-score">
                        {item.confidence_label}
                      </span>
                    </span>

                    <span className="comparable-caption">{item.caption}</span>

                    <span className="comparable-author">{item.author}</span>

                    <span className="comparable-hashtags">
                      {item.hashtags.map((tag) => (
                        <span className="comparable-tag" key={`${item.id}-${tag}`}>
                          {tag}
                        </span>
                      ))}
                    </span>

                    <span className="comparable-metrics-inline">
                      <span>Views: {formatCompactNumber(item.metrics.views)}</span>
                      <span>Likes: {formatCompactNumber(item.metrics.likes)}</span>
                      <span>
                        Comments: {formatCompactNumber(item.metrics.comments_count)}
                      </span>
                      <span>Shares: {formatCompactNumber(item.metrics.shares)}</span>
                    </span>

                    <span className="comparable-hover-details">
                      <span>Likes: {item.metrics.likes.toLocaleString("en-US")}</span>
                      <span>
                        Comments: {item.metrics.comments_count.toLocaleString("en-US")}
                      </span>
                      <span>Shares: {item.metrics.shares.toLocaleString("en-US")}</span>
                      <span>Engagement: {item.metrics.engagement_rate}</span>
                      <span>Support: {item.support_level}</span>
                    </span>
                  </span>
                </button>

                <div className="comparable-feedback-row">
                  <button
                    type="button"
                    className="report-ghost-action"
                    onClick={() => props.onMarkRelevant(item, "relevant")}
                  >
                    {props.feedbackState[item.id] === "relevant" ? "Relevant saved" : "Mark relevant"}
                  </button>
                  <button
                    type="button"
                    className="report-ghost-action"
                    onClick={() => props.onMarkRelevant(item, "not_relevant")}
                  >
                    {props.feedbackState[item.id] === "not_relevant"
                      ? "Marked not relevant"
                      : "Mark not relevant"}
                  </button>
                  <button
                    type="button"
                    className="report-ghost-action"
                    onClick={() => props.onSaveComparable(item)}
                  >
                    {props.feedbackState[item.id] === "saved" ? "Saved" : "Save comparable"}
                  </button>
                </div>

                <div className="comparable-evidence-block">
                  <p>{item.why_this_was_chosen}</p>
                  <p>
                    Reasons: {item.ranking_reasons.length > 0 ? item.ranking_reasons.join(", ") : "Not available"}
                  </p>
                </div>

                {hasVideoUrl ? (
                  <a
                    href={item.video_url}
                    target="_blank"
                    rel="noreferrer"
                    className="comparable-open-link"
                    aria-label="Open this video on TikTok"
                    onClick={() => props.onOpenVideo(item)}
                  >
                    View on TikTok
                  </a>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
