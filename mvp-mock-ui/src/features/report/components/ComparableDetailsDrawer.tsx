import { CloseIcon } from "./ReportIcons";
import type { ComparableItem } from "../types";
import { ComparableThumbnailImage } from "./ComparableThumbnailImage";

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
                <h4>Comparable details</h4>
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

              <section className="report-drawer-block">
                <h5>Full caption</h5>
                <p>{item.caption}</p>
              </section>

              <section className="report-drawer-block">
                <h5>Metrics</h5>
                <ul>
                  <li>Views: {formatNumber(item.metrics.views)}</li>
                  <li>Likes: {formatNumber(item.metrics.likes)}</li>
                  <li>Comments: {formatNumber(item.metrics.comments_count)}</li>
                  <li>Shares: {formatNumber(item.metrics.shares)}</li>
                  <li>Engagement rate: {item.metrics.engagement_rate}</li>
                </ul>
              </section>

              <section className="report-drawer-block">
                <h5>Why this is similar</h5>
                <ul>
                  {item.matched_keywords.map((keyword) => (
                    <li key={keyword}>{keyword}</li>
                  ))}
                </ul>
              </section>

              <section className="report-drawer-block">
                <h5>Observations</h5>
                <ul>
                  {item.observations.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </section>
            </div>
          </>
        ) : null}
      </aside>
    </>
  );
}
