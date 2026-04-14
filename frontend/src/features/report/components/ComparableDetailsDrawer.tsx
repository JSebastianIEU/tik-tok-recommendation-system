import { CloseIcon } from "./ReportIcons";
import type { ComparableItem } from "../types";
import { ComparableThumbnailImage } from "./ComparableThumbnailImage";
import { ScoreComponentsChart } from "./ScoreBar";

interface ComparableDetailsDrawerProps {
  item: ComparableItem | null;
  onClose: () => void;
}

function formatNumber(value: number): string {
  return value.toLocaleString("en-US");
}

export function ComparableDetailsDrawer(
  props: ComparableDetailsDrawerProps
): JSX.Element {
  const { item, onClose } = props;
  const isOpen = Boolean(item);

  return (
    <>
      <button
        type="button"
        className={`report-drawer-overlay ${isOpen ? "report-drawer-open" : ""}`}
        onClick={onClose}
        aria-label="Close comparable details"
      />

      <aside
        className={`report-drawer ${isOpen ? "report-drawer-open" : ""}`}
        aria-hidden={!isOpen}
      >
        {item ? (
          <>
            <header className="report-drawer-header">
              <div>
                <h4>Comparable Details</h4>
                <p className="drawer-similarity">
                  Similarity: {Math.round(item.similarity * 100)}% · {item.support_level} support
                </p>
              </div>

              <button
                type="button"
                className="report-drawer-close"
                onClick={onClose}
                aria-label="Close"
              >
                <CloseIcon className="report-action-icon" />
              </button>
            </header>

            <div className="report-drawer-body">
              <ComparableThumbnailImage
                className="report-drawer-thumb-image"
                thumbnailUrl={item.thumbnail_url}
                videoUrl={item.video_url}
                alt="Comparable thumbnail"
                fallbackClassName="report-drawer-thumb"
              />

              {item.video_url ? (
                <a
                  href={item.video_url}
                  target="_blank"
                  rel="noreferrer"
                  className="report-drawer-video-link"
                >
                  Open on TikTok
                </a>
              ) : null}

              <section className="drawer-block">
                <h5>Caption</h5>
                <p>{item.caption}</p>
                <p className="drawer-author">@{item.author.replace(/^@+/, "")}</p>
              </section>

              <section className="drawer-block">
                <h5>Performance Metrics</h5>
                <div className="drawer-metrics-grid">
                  <div className="drawer-metric">
                    <span className="drawer-metric-val">{formatNumber(item.metrics.views)}</span>
                    <span className="drawer-metric-lbl">Views</span>
                  </div>
                  <div className="drawer-metric">
                    <span className="drawer-metric-val">{formatNumber(item.metrics.likes)}</span>
                    <span className="drawer-metric-lbl">Likes</span>
                  </div>
                  <div className="drawer-metric">
                    <span className="drawer-metric-val">{formatNumber(item.metrics.comments_count)}</span>
                    <span className="drawer-metric-lbl">Comments</span>
                  </div>
                  <div className="drawer-metric">
                    <span className="drawer-metric-val">{formatNumber(item.metrics.shares)}</span>
                    <span className="drawer-metric-lbl">Shares</span>
                  </div>
                  <div className="drawer-metric drawer-metric-wide">
                    <span className="drawer-metric-val">{item.metrics.engagement_rate}</span>
                    <span className="drawer-metric-lbl">Engagement Rate</span>
                  </div>
                </div>
              </section>

              <section className="drawer-block">
                <h5>Score Breakdown</h5>
                <ScoreComponentsChart components={item.score_components} />
              </section>

              {item.hashtags.length > 0 && (
                <section className="drawer-block">
                  <h5>Hashtags</h5>
                  <div className="drawer-tags">
                    {item.hashtags.map((tag) => (
                      <span className="drawer-tag" key={tag}>{tag}</span>
                    ))}
                  </div>
                </section>
              )}

              {item.ranking_reasons.length > 0 && (
                <section className="drawer-block">
                  <h5>Selection Reasons</h5>
                  <div className="drawer-reasons">
                    {item.ranking_reasons.map((reason, i) => (
                      <span className="drawer-reason" key={i}>
                        {reason.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                </section>
              )}

              {item.retrieval_branches.length > 0 && (
                <section className="drawer-block">
                  <h5>Retrieval Branches</h5>
                  <div className="drawer-reasons">
                    {item.retrieval_branches.map((branch, i) => (
                      <span className="drawer-reason" key={i}>
                        {branch.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </>
        ) : null}
      </aside>
    </>
  );
}
