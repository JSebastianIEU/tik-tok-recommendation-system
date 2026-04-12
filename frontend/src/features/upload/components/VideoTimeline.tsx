import { useState } from "react";
import type { FrameTimelineEntry } from "../../../services/contracts/models";

interface VideoTimelineProps {
  timeline: FrameTimelineEntry[];
  duration: number;
}

function formatTimestamp(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function VideoTimeline(props: VideoTimelineProps): JSX.Element {
  const { timeline, duration } = props;
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!timeline || timeline.length === 0) {
    return <></>;
  }

  const maxRelevance = Math.max(...timeline.map((e) => e.relevance_score), 0.01);

  return (
    <section className="glass-card video-timeline-panel">
      <h3 className="panel-title">video timeline</h3>
      <p className="timeline-subtitle">
        {timeline.length} frames &middot; {formatTimestamp(duration)} total
      </p>

      {/* Relevance bar chart */}
      <div className="timeline-bar-chart">
        {timeline.map((entry, idx) => {
          const barHeight = (entry.relevance_score / maxRelevance) * 100;
          const isHovered = hoveredIdx === idx;
          return (
            <div
              key={idx}
              className={`timeline-bar-col ${isHovered ? "hovered" : ""}`}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              <div
                className={`timeline-bar ${entry.is_scene_change ? "scene-change" : ""}`}
                style={{ height: `${barHeight}%` }}
              />
              {entry.is_scene_change && (
                <div className="scene-marker" title="Scene change" />
              )}
            </div>
          );
        })}
      </div>

      {/* Thumbnail strip */}
      <div className="timeline-thumb-strip">
        {timeline.map((entry, idx) => (
          <div
            key={idx}
            className={`timeline-thumb-item ${hoveredIdx === idx ? "hovered" : ""}`}
            onMouseEnter={() => setHoveredIdx(idx)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            <img
              src={`data:image/jpeg;base64,${entry.thumbnail_b64}`}
              alt={`Frame at ${formatTimestamp(entry.timestamp_sec)}`}
              className="timeline-thumb-img"
            />
            <span className="timeline-thumb-time">
              {formatTimestamp(entry.timestamp_sec)}
            </span>
            {entry.face_count > 0 && (
              <span className="timeline-badge face-badge" title="Faces detected">
                {entry.face_count}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Detail tooltip for hovered frame */}
      {hoveredIdx !== null && timeline[hoveredIdx] && (
        <div className="timeline-detail">
          <div className="timeline-detail-row">
            <span className="timeline-detail-label">Time</span>
            <span>{formatTimestamp(timeline[hoveredIdx].timestamp_sec)}</span>
          </div>
          <div className="timeline-detail-row">
            <span className="timeline-detail-label">Relevance</span>
            <span>{Math.round(timeline[hoveredIdx].relevance_score * 100)}%</span>
          </div>
          <div className="timeline-detail-row">
            <span className="timeline-detail-label">Motion</span>
            <span>{(timeline[hoveredIdx].motion_score * 100).toFixed(1)}%</span>
          </div>
          {timeline[hoveredIdx].face_count > 0 && (
            <div className="timeline-detail-row">
              <span className="timeline-detail-label">Faces</span>
              <span>{timeline[hoveredIdx].face_count}</span>
            </div>
          )}
          {timeline[hoveredIdx].is_scene_change && (
            <div className="timeline-detail-row">
              <span className="timeline-detail-tag">Scene Change</span>
            </div>
          )}
          {timeline[hoveredIdx].ocr_text && (
            <div className="timeline-detail-row">
              <span className="timeline-detail-label">Text</span>
              <span className="timeline-detail-ocr">{timeline[hoveredIdx].ocr_text}</span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
