import type { ComparableItem } from "../types";
import { ComparableThumbnailImage } from "./ComparableThumbnailImage";
import { ScoreComponentsChart } from "./ScoreBar";

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

function getSupportColor(level: string): string {
  switch (level) {
    case "full": return "rgba(34,197,94,0.9)";
    case "partial": return "rgba(251,191,36,0.9)";
    case "low": return "rgba(239,68,68,0.9)";
    default: return "rgba(148,163,184,0.7)";
  }
}

function cleanAuthor(author: string): string {
  // Remove leading @ symbols — backend already adds one, avoid @@
  return author.replace(/^@+/, "");
}

export function ComparablesSection(props: ComparablesSectionProps): JSX.Element {
  const { items, selectedComparableId, onSelectComparable } = props;

  return (
    <section
      className="report-section report-comparables-section"
      aria-labelledby="comparables-title"
    >
      <div className="report-section-head">
        <h3 id="comparables-title">Comparable Videos</h3>
        <p>Top matches from the recommendation engine</p>
      </div>

      <div className="comparable-grid" role="list">
        {items.map((item, itemIndex) => {
          const isActive = selectedComparableId === item.id;
          const hasVideoUrl = item.video_url.trim().length > 0;

          return (
            <article
              className={`comp-card ${isActive ? "comp-card-active" : ""}`}
              key={`${item.id}-${itemIndex}`}
              role="listitem"
            >
              <div className="comp-card-header">
                <div className="comp-card-rank">#{itemIndex + 1}</div>
                <div className="comp-card-similarity-ring">
                  <svg viewBox="0 0 36 36" className="comp-ring-svg">
                    <path
                      className="comp-ring-bg"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path
                      className="comp-ring-fill"
                      strokeDasharray={`${Math.round(item.similarity * 100)}, 100`}
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                  </svg>
                  <span className="comp-ring-label">{formatSimilarity(item.similarity)}</span>
                </div>
                <span
                  className="comp-support-badge"
                  style={{ color: getSupportColor(item.support_level) }}
                >
                  {item.support_level}
                </span>
              </div>

              {hasVideoUrl ? (
                <a
                  href={item.video_url}
                  target="_blank"
                  rel="noreferrer"
                  className="comp-thumb-link"
                  aria-label="Open comparable video on TikTok"
                  onClick={() => props.onOpenVideo(item)}
                >
                  <ComparableThumbnailImage
                    className="comp-thumb-img"
                    thumbnailUrl={item.thumbnail_url}
                    videoUrl={item.video_url}
                    alt="Comparable thumbnail"
                    fallbackClassName="comp-thumb-fallback"
                  />
                  <span className="comp-thumb-play">Open on TikTok</span>
                </a>
              ) : (
                <span className="comp-thumb-fallback" aria-hidden="true" />
              )}

              <button
                type="button"
                className="comp-card-body"
                onClick={() => onSelectComparable(item.id)}
              >
                <p className="comp-caption">{item.caption}</p>
                <p className="comp-author">@{cleanAuthor(item.author)}</p>

                <div className="comp-stats-row">
                  <span className="comp-stat">
                    <span className="comp-stat-icon">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                    </span>
                    <span className="comp-stat-val">{formatCompactNumber(item.metrics.views)}</span>
                    <span className="comp-stat-lbl">views</span>
                  </span>
                  <span className="comp-stat">
                    <span className="comp-stat-icon">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
                    </span>
                    <span className="comp-stat-val">{formatCompactNumber(item.metrics.likes)}</span>
                    <span className="comp-stat-lbl">likes</span>
                  </span>
                  <span className="comp-stat">
                    <span className="comp-stat-icon">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    </span>
                    <span className="comp-stat-val">{formatCompactNumber(item.metrics.comments_count)}</span>
                    <span className="comp-stat-lbl">comments</span>
                  </span>
                  <span className="comp-stat">
                    <span className="comp-stat-icon">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
                    </span>
                    <span className="comp-stat-val">{formatCompactNumber(item.metrics.shares)}</span>
                    <span className="comp-stat-lbl">shares</span>
                  </span>
                </div>

                <div className="comp-engagement-row">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                  <span className="comp-engagement-val">{item.metrics.engagement_rate}</span>
                  <span className="comp-engagement-lbl">engagement rate</span>
                </div>

                <div className="comp-tags-row">
                  {item.hashtags.slice(0, 6).map((tag) => (
                    <span className="comp-tag" key={`${item.id}-${tag}`}>
                      {tag}
                    </span>
                  ))}
                  {item.hashtags.length > 6 && (
                    <span className="comp-tag comp-tag-more">+{item.hashtags.length - 6}</span>
                  )}
                </div>

                <div className="comp-score-section">
                  <h5>Score Breakdown</h5>
                  <ScoreComponentsChart components={item.score_components} compact />
                </div>
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}
