import type { ComparableItem } from "../types";
import { ComparableThumbnailImage } from "./ComparableThumbnailImage";

interface ComparablesSectionProps {
  items: ComparableItem[];
  selectedComparableId: string | null;
  onSelectComparable: (id: string) => void;
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
                    </span>
                  </span>
                </button>

                {hasVideoUrl ? (
                  <a
                    href={item.video_url}
                    target="_blank"
                    rel="noreferrer"
                    className="comparable-open-link"
                    aria-label="Open this video on TikTok"
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
