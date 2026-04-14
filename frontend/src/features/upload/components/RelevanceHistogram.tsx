import { useMemo, useState } from "react";
import type { FrameTimelineEntry } from "../../../services/api/videoAnalysisApi";

interface RelevanceHistogramProps {
  frames: FrameTimelineEntry[];
  durationSeconds: number;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const BAR_GAP = 2;
const CHART_HEIGHT = 120;
const LABEL_HEIGHT = 20;
const SVG_PADDING = { top: 8, right: 12, bottom: LABEL_HEIGHT + 4, left: 12 };

export function RelevanceHistogram(
  props: RelevanceHistogramProps
): JSX.Element {
  const { frames } = props;
  const [hoveredBar, setHoveredBar] = useState<number | null>(null);

  const chartData = useMemo(() => {
    if (frames.length === 0) return [];
    return frames.map((f) => ({
      timestamp: f.timestamp_sec,
      relevance: f.relevance_score,
      isSceneChange: f.is_scene_change,
      caption: f.caption,
    }));
  }, [frames]);

  if (chartData.length === 0) {
    return <div className="relevance-histogram-empty">No data</div>;
  }

  const totalWidth = Math.max(chartData.length * 24, 300);
  const svgWidth = totalWidth + SVG_PADDING.left + SVG_PADDING.right;
  const svgHeight = CHART_HEIGHT + SVG_PADDING.top + SVG_PADDING.bottom;
  const chartWidth = totalWidth;
  const barWidth = Math.max(
    (chartWidth - (chartData.length - 1) * BAR_GAP) / chartData.length,
    6
  );

  const maxRelevance = Math.max(...chartData.map((d) => d.relevance), 0.01);

  return (
    <div className="relevance-histogram-container">
      <div className="relevance-histogram-header">
        <span className="relevance-histogram-title">
          Relevance Map
        </span>
        <span className="relevance-histogram-subtitle">
          Higher bars = more relevant to your description
        </span>
      </div>

      <div className="relevance-histogram-scroll">
        <svg
          width={svgWidth}
          height={svgHeight}
          className="relevance-histogram-svg"
        >
          {chartData.map((d, i) => {
            const x = SVG_PADDING.left + i * (barWidth + BAR_GAP);
            const barHeight =
              (d.relevance / maxRelevance) * (CHART_HEIGHT - SVG_PADDING.top);
            const y =
              SVG_PADDING.top + (CHART_HEIGHT - SVG_PADDING.top - barHeight);
            const isHovered = hoveredBar === i;

            const hue = d.relevance > 0.7 ? 280 : d.relevance > 0.4 ? 250 : 220;
            const alpha = isHovered ? 1 : 0.75;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredBar(i)}
                onMouseLeave={() => setHoveredBar(null)}
              >
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx={3}
                  fill={`hsla(${hue}, 70%, 65%, ${alpha})`}
                  stroke={
                    d.isSceneChange
                      ? "rgba(255, 200, 100, 0.9)"
                      : "transparent"
                  }
                  strokeWidth={d.isSceneChange ? 2 : 0}
                  className="histogram-bar"
                />

                {/* Scene change triangle marker */}
                {d.isSceneChange && (
                  <polygon
                    points={`${x + barWidth / 2 - 4},${y - 2} ${x + barWidth / 2 + 4},${y - 2} ${x + barWidth / 2},${y - 8}`}
                    fill="rgba(255, 200, 100, 0.9)"
                  />
                )}

                {/* Timestamp labels (every 3rd) */}
                {i % 3 === 0 && (
                  <text
                    x={x + barWidth / 2}
                    y={svgHeight - 4}
                    textAnchor="middle"
                    className="histogram-label"
                  >
                    {formatTimestamp(d.timestamp)}
                  </text>
                )}

                {/* Hover tooltip */}
                {isHovered && (
                  <g>
                    <rect
                      x={Math.max(0, x - 60)}
                      y={Math.max(0, y - 50)}
                      width={150}
                      height={42}
                      rx={6}
                      fill="rgba(20, 10, 40, 0.92)"
                      stroke="rgba(180, 140, 255, 0.3)"
                    />
                    <text
                      x={Math.max(0, x - 60) + 8}
                      y={Math.max(0, y - 50) + 16}
                      className="histogram-tooltip-text"
                    >
                      {Math.round(d.relevance * 100)}% relevant
                    </text>
                    <text
                      x={Math.max(0, x - 60) + 8}
                      y={Math.max(0, y - 50) + 32}
                      className="histogram-tooltip-sub"
                    >
                      {formatTimestamp(d.timestamp)}
                      {d.isSceneChange ? " | Scene cut" : ""}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="relevance-histogram-legend">
        <span className="legend-item">
          <span
            className="legend-dot"
            style={{ background: "hsla(280, 70%, 65%, 0.9)" }}
          />
          High relevance
        </span>
        <span className="legend-item">
          <span
            className="legend-dot"
            style={{ background: "hsla(220, 70%, 65%, 0.9)" }}
          />
          Low relevance
        </span>
        <span className="legend-item">
          <span
            className="legend-dot"
            style={{ background: "rgba(255, 200, 100, 0.9)" }}
          />
          Scene change
        </span>
      </div>
    </div>
  );
}
