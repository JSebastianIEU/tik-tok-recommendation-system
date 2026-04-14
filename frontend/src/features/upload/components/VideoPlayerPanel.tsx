import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  FrameTimelineEntry,
  VideoAnalysisResult
} from "../../../services/contracts/models";

type Tab = "timeline" | "transcript" | "frames";

interface VideoPlayerPanelProps {
  videoUrl: string | null;
  fileName: string | null;
  analysis: VideoAnalysisResult | null;
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function describeFrame(entry: FrameTimelineEntry, idx: number, total: number): string {
  const parts: string[] = [];
  if (idx === 0) parts.push("Opening shot");
  else if (idx === total - 1) parts.push("Closing shot");
  if (entry.is_scene_change) parts.push("Scene change");
  if (entry.face_count === 1) parts.push("1 person visible");
  else if (entry.face_count > 1) parts.push(`${entry.face_count} people visible`);
  if (entry.motion_score > 0.15) parts.push("High motion");
  else if (entry.motion_score > 0.05) parts.push("Moderate motion");
  else parts.push("Still frame");
  if (entry.ocr_text) parts.push(`Text: "${entry.ocr_text}"`);
  return parts.join(" · ");
}

/* ── Key-frame picker: top N by relevance, de-duplicated by time gap ── */
function pickKeyFrames(
  timeline: FrameTimelineEntry[],
  count: number
): FrameTimelineEntry[] {
  if (timeline.length <= count) return [...timeline];
  const sorted = [...timeline].sort(
    (a, b) => b.relevance_score - a.relevance_score
  );
  const picked: FrameTimelineEntry[] = [];
  for (const frame of sorted) {
    if (picked.length >= count) break;
    const tooClose = picked.some(
      (p) => Math.abs(p.timestamp_sec - frame.timestamp_sec) < 1.5
    );
    if (!tooClose) picked.push(frame);
  }
  return picked.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
}

/* ── SVG line-graph path from timeline data ── */
function buildGraphPath(
  timeline: FrameTimelineEntry[],
  width: number,
  height: number,
  field: "relevance_score" | "motion_score"
): string {
  if (timeline.length === 0) return "";
  const maxVal = Math.max(...timeline.map((e) => e[field]), 0.01);
  const pad = 2;
  const usableH = height - pad * 2;
  const usableW = width - pad * 2;
  return timeline
    .map((entry, idx) => {
      const x = pad + (idx / Math.max(timeline.length - 1, 1)) * usableW;
      const y = pad + usableH - (entry[field] / maxVal) * usableH;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function buildAreaPath(
  timeline: FrameTimelineEntry[],
  width: number,
  height: number,
  field: "relevance_score" | "motion_score"
): string {
  const line = buildGraphPath(timeline, width, height, field);
  if (!line) return "";
  const pad = 2;
  const usableW = width - pad * 2;
  const lastX = pad + usableW;
  return `${line} L${lastX.toFixed(1)},${height} L${pad},${height} Z`;
}

export function VideoPlayerPanel(props: VideoPlayerPanelProps): JSX.Element {
  const { videoUrl, fileName, analysis } = props;

  const videoRef = useRef<HTMLVideoElement>(null);
  const graphRef = useRef<SVGSVGElement>(null);
  const [previewError, setPreviewError] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [activeTab, setActiveTab] = useState<Tab>("timeline");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const timeline = analysis?.timeline ?? [];
  const transcript = analysis?.transcript ?? "";
  const videoCaption = analysis?.video_caption ?? "";
  const ocrText = analysis?.ocr_text ?? "";
  const keyFrames = useMemo(() => pickKeyFrames(timeline, 8), [timeline]);

  /* ── Video setup ── */
  useEffect(() => {
    setPreviewError(false);
  }, [videoUrl]);

  useEffect(() => {
    if (!videoUrl || !videoRef.current) return;
    const el = videoRef.current;
    void el.play().catch(() => {
      setPreviewError(true);
      setIsPlaying(false);
    });
  }, [videoUrl]);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  }, []);

  const seekTo = useCallback((sec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      setCurrentTime(sec);
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      void videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  }, []);

  const toggleMute = useCallback(() => {
    if (!videoRef.current) return;
    videoRef.current.muted = !videoRef.current.muted;
    setIsMuted(videoRef.current.muted);
  }, []);

  /* ── Which timeline entry is "active" (closest to current playback) ── */
  const activeBarIdx = useMemo(() => {
    if (timeline.length === 0) return -1;
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < timeline.length; i++) {
      const dist = Math.abs(timeline[i].timestamp_sec - currentTime);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }, [timeline, currentTime]);

  /* ── Scene-change markers positions ── */
  const sceneMarkers = useMemo(() => {
    if (timeline.length < 2) return [];
    const total = timeline.length - 1;
    return timeline
      .map((e, i) => ({ idx: i, pct: (i / total) * 100, entry: e }))
      .filter((m) => m.entry.is_scene_change);
  }, [timeline]);

  const hasTimeline = timeline.length > 0;
  const hasTranscript = transcript.trim().length > 0 || videoCaption.trim().length > 0 || ocrText.trim().length > 0;
  const hasFrames = keyFrames.length > 0;

  const inspectedIdx = selectedIdx ?? hoveredIdx;
  const inspectedEntry = inspectedIdx !== null ? timeline[inspectedIdx] : null;

  /* ── Graph click handler ── */
  const handleGraphClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (timeline.length === 0) return;
      const svg = graphRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = x / rect.width;
      const idx = Math.round(pct * (timeline.length - 1));
      const clamped = Math.max(0, Math.min(timeline.length - 1, idx));
      setSelectedIdx(clamped);
      seekTo(timeline[clamped].timestamp_sec);
    },
    [timeline, seekTo]
  );

  const handleGraphMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (timeline.length === 0) return;
      const svg = graphRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = x / rect.width;
      const idx = Math.round(pct * (timeline.length - 1));
      setHoveredIdx(Math.max(0, Math.min(timeline.length - 1, idx)));
    },
    [timeline]
  );

  const GRAPH_W = 400;
  const GRAPH_H = 56;

  return (
    <section className="glass-card vp-panel">
      {/* ── Video with controls overlay ── */}
      <div className="vp-video-wrap">
        {videoUrl && !previewError ? (
          <video
            ref={videoRef}
            className="vp-video"
            src={videoUrl}
            muted
            playsInline
            autoPlay
            loop
            preload="metadata"
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onError={() => setPreviewError(true)}
          />
        ) : (
          <div className="vp-fallback">
            {previewError
              ? "Video could not be played"
              : "Preview not available"}
          </div>
        )}

        {/* Controls overlay */}
        <div className="vp-controls">
          <button type="button" className="vp-ctrl-btn" onClick={togglePlay} title={isPlaying ? "Pause" : "Play"}>
            {isPlaying ? (
              <svg viewBox="0 0 24 24" width="20" height="20">
                <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor" />
                <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="20" height="20">
                <polygon points="6,4 20,12 6,20" fill="currentColor" />
              </svg>
            )}
          </button>
          <span className="vp-time-display">
            {fmt(currentTime)} / {fmt(duration)}
          </span>
          <button type="button" className="vp-ctrl-btn" onClick={toggleMute} title={isMuted ? "Unmute" : "Mute"}>
            {isMuted ? (
              <svg viewBox="0 0 24 24" width="18" height="18">
                <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" />
                <line x1="18" y1="9" x2="22" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <line x1="22" y1="9" x2="18" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="18" height="18">
                <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" />
                <path d="M15.5 8.5a5 5 0 010 7M19 6a9 9 0 010 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* ── Line-graph scrubber ── */}
      {hasTimeline && (
        <div className="vp-graph-wrap">
          <svg
            ref={graphRef}
            className="vp-graph-svg"
            viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`}
            preserveAspectRatio="none"
            onClick={handleGraphClick}
            onMouseMove={handleGraphMove}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            {/* Relevance area fill */}
            <path
              d={buildAreaPath(timeline, GRAPH_W, GRAPH_H, "relevance_score")}
              fill="rgba(167,139,250,0.15)"
            />
            {/* Motion area fill */}
            <path
              d={buildAreaPath(timeline, GRAPH_W, GRAPH_H, "motion_score")}
              fill="rgba(56,189,248,0.1)"
            />
            {/* Relevance line */}
            <path
              d={buildGraphPath(timeline, GRAPH_W, GRAPH_H, "relevance_score")}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
            {/* Motion line */}
            <path
              d={buildGraphPath(timeline, GRAPH_W, GRAPH_H, "motion_score")}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="1.5"
              strokeDasharray="4 3"
              vectorEffect="non-scaling-stroke"
            />
            {/* Scene-change markers */}
            {sceneMarkers.map((m) => {
              const x = 2 + (m.idx / Math.max(timeline.length - 1, 1)) * (GRAPH_W - 4);
              return (
                <line
                  key={m.idx}
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={GRAPH_H}
                  stroke="rgba(251,191,36,0.6)"
                  strokeWidth="1"
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
            {/* Playhead position */}
            {duration > 0 && (
              <line
                x1={2 + (currentTime / duration) * (GRAPH_W - 4)}
                y1={0}
                x2={2 + (currentTime / duration) * (GRAPH_W - 4)}
                y2={GRAPH_H}
                stroke="rgba(255,255,255,0.7)"
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {/* Hovered dot */}
            {hoveredIdx !== null && (
              <circle
                cx={2 + (hoveredIdx / Math.max(timeline.length - 1, 1)) * (GRAPH_W - 4)}
                cy={
                  2 +
                  (GRAPH_H - 4) -
                  (timeline[hoveredIdx].relevance_score /
                    Math.max(...timeline.map((e) => e.relevance_score), 0.01)) *
                    (GRAPH_H - 4)
                }
                r="4"
                fill="#a78bfa"
                stroke="#fff"
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
          {/* Legend */}
          <div className="vp-graph-legend">
            <span className="vp-legend-item"><span className="vp-legend-dot" style={{ background: "#a78bfa" }} />Relevance</span>
            <span className="vp-legend-item"><span className="vp-legend-dot" style={{ background: "#38bdf8" }} />Motion</span>
            <span className="vp-legend-item"><span className="vp-legend-dot" style={{ background: "#fbbf24" }} />Scene cut</span>
          </div>
        </div>
      )}

      {/* ── Tabs ── */}
      <div className="vp-tabs">
        {hasTimeline && (
          <button
            type="button"
            className={`vp-tab ${activeTab === "timeline" ? "active" : ""}`}
            onClick={() => setActiveTab("timeline")}
          >
            <svg viewBox="0 0 24 24" className="vp-tab-icon">
              <polyline points="4,18 8,10 12,14 16,6 20,12" />
            </svg>
            Timeline
          </button>
        )}
        <button
          type="button"
          className={`vp-tab ${activeTab === "transcript" ? "active" : ""}`}
          onClick={() => setActiveTab("transcript")}
        >
          <svg viewBox="0 0 24 24" className="vp-tab-icon">
            <path d="M4 6H20M4 10H16M4 14H20M4 18H12" />
          </svg>
          Info
        </button>
        {hasFrames && (
          <button
            type="button"
            className={`vp-tab ${activeTab === "frames" ? "active" : ""}`}
            onClick={() => setActiveTab("frames")}
          >
            <svg viewBox="0 0 24 24" className="vp-tab-icon">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
            Key Frames
          </button>
        )}
      </div>

      {/* ── Tab content ── */}
      <div className="vp-tab-body">
        {/* ── Timeline tab ── */}
        {activeTab === "timeline" && hasTimeline && (
          <div className="vp-tl-content">
            {/* Compact thumbnail strip */}
            <div className="vp-thumb-strip">
              {timeline.map((entry, idx) => {
                const isActive = idx === activeBarIdx;
                const isSelected = idx === selectedIdx;
                const isHov = idx === hoveredIdx;
                return (
                  <button
                    key={idx}
                    type="button"
                    className={[
                      "vp-thumb",
                      isActive ? "active" : "",
                      isSelected ? "selected" : "",
                      isHov ? "hovered" : ""
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => {
                      setSelectedIdx(idx);
                      seekTo(entry.timestamp_sec);
                    }}
                    onMouseEnter={() => setHoveredIdx(idx)}
                    onMouseLeave={() => setHoveredIdx(null)}
                  >
                    <img
                      src={`data:image/jpeg;base64,${entry.thumbnail_b64}`}
                      alt={`Frame ${fmt(entry.timestamp_sec)}`}
                      className="vp-thumb-img"
                    />
                    <span className="vp-thumb-time">{fmt(entry.timestamp_sec)}</span>
                    {entry.is_scene_change && (
                      <span className="vp-thumb-badge scene">cut</span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Selected / hovered frame detail card */}
            {inspectedEntry && inspectedIdx !== null && (
              <div className="vp-frame-detail-card">
                <div className="vp-fdc-header">
                  <span className="vp-fdc-time">{fmt(inspectedEntry.timestamp_sec)}</span>
                  <span className="vp-fdc-relevance">
                    Relevance {Math.round(inspectedEntry.relevance_score * 100)}%
                  </span>
                </div>
                <p className="vp-fdc-description">
                  {describeFrame(inspectedEntry, inspectedIdx, timeline.length)}
                </p>
                <div className="vp-fdc-metrics">
                  <span>
                    <span className="vp-fdc-metric-label">Motion</span>
                    <span className="vp-fdc-metric-bar">
                      <span
                        className="vp-fdc-metric-fill motion"
                        style={{ width: `${Math.min(inspectedEntry.motion_score * 500, 100)}%` }}
                      />
                    </span>
                  </span>
                  <span>
                    <span className="vp-fdc-metric-label">Relevance</span>
                    <span className="vp-fdc-metric-bar">
                      <span
                        className="vp-fdc-metric-fill relevance"
                        style={{ width: `${inspectedEntry.relevance_score * 100}%` }}
                      />
                    </span>
                  </span>
                  {inspectedEntry.face_count > 0 && (
                    <span className="vp-fdc-faces">
                      {inspectedEntry.face_count} face{inspectedEntry.face_count > 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Info / Transcript tab ── */}
        {activeTab === "transcript" && (
          <div className="vp-transcript">
            {/* AI-generated caption */}
            {videoCaption && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">AI Video Description</h5>
                <p className="vp-info-text">{videoCaption}</p>
              </div>
            )}

            {/* Transcript */}
            {transcript && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">Transcript</h5>
                <p className="vp-info-text">{transcript}</p>
              </div>
            )}

            {/* OCR text */}
            {ocrText && ocrText.trim().length > 0 && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">On-Screen Text (OCR)</h5>
                <p className="vp-info-text">{ocrText}</p>
              </div>
            )}

            {/* Visual features */}
            {analysis?.visual_features && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">Visual Features</h5>
                <div className="vp-visual-features">
                  <span className="vp-vf-item">
                    <span className="vp-vf-key">Resolution</span>
                    <span className="vp-vf-val">{analysis.visual_features.resolution}</span>
                  </span>
                  <span className="vp-vf-item">
                    <span className="vp-vf-key">Aspect</span>
                    <span className="vp-vf-val">{analysis.visual_features.aspect_ratio}</span>
                  </span>
                  <span className="vp-vf-item">
                    <span className="vp-vf-key">Brightness</span>
                    <span className="vp-vf-val">{Math.round((analysis.visual_features.avg_brightness ?? 0) * 100)}%</span>
                  </span>
                  {analysis.visual_features.face_count !== undefined && (
                    <span className="vp-vf-item">
                      <span className="vp-vf-key">Faces</span>
                      <span className="vp-vf-val">{analysis.visual_features.face_count}</span>
                    </span>
                  )}
                  {analysis.visual_features.dominant_colors && (
                    <span className="vp-vf-item">
                      <span className="vp-vf-key">Colors</span>
                      <span className="vp-vf-colors">
                        {analysis.visual_features.dominant_colors.map((c, i) => (
                          <span key={i} className="vp-color-dot" style={{ background: c }} title={c} />
                        ))}
                      </span>
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Duration */}
            {analysis?.duration_seconds && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">Duration</h5>
                <p className="vp-info-text">{fmt(analysis.duration_seconds)} ({analysis.duration_seconds.toFixed(1)}s)</p>
              </div>
            )}

            {/* Detected language */}
            {analysis?.detected_language && (
              <div className="vp-info-block">
                <h5 className="vp-info-label">Language</h5>
                <p className="vp-info-text">{analysis.detected_language.toUpperCase()}</p>
              </div>
            )}

            {/* Empty state */}
            {!videoCaption && !transcript && !ocrText && !analysis?.visual_features && (
              <p className="vp-empty">No transcript or description data available for this video.</p>
            )}
          </div>
        )}

        {/* ── Key Frames tab ── */}
        {activeTab === "frames" && hasFrames && (
          <div className="vp-frames-grid">
            {keyFrames.map((frame, idx) => (
              <button
                key={idx}
                type="button"
                className="vp-frame-card"
                onClick={() => seekTo(frame.timestamp_sec)}
              >
                <img
                  src={`data:image/jpeg;base64,${frame.thumbnail_b64}`}
                  alt={`Key frame ${fmt(frame.timestamp_sec)}`}
                  className="vp-frame-img"
                />
                <div className="vp-frame-overlay">
                  <span className="vp-frame-time">{fmt(frame.timestamp_sec)}</span>
                  <span className="vp-frame-score">
                    {Math.round(frame.relevance_score * 100)}%
                  </span>
                </div>
                <div className="vp-frame-tags">
                  {frame.is_scene_change && <span className="vp-ftag scene">Scene Cut</span>}
                  {frame.face_count > 0 && (
                    <span className="vp-ftag face">{frame.face_count} face{frame.face_count > 1 ? "s" : ""}</span>
                  )}
                  {frame.motion_score > 0.05 && (
                    <span className="vp-ftag motion">Motion {Math.round(frame.motion_score * 100)}%</span>
                  )}
                  {frame.ocr_text && (
                    <span className="vp-ftag ocr" title={frame.ocr_text}>"{frame.ocr_text}"</span>
                  )}
                </div>
                <p className="vp-frame-desc">
                  {describeFrame(frame, timeline.indexOf(frame), timeline.length)}
                </p>
              </button>
            ))}
          </div>
        )}

        {/* ── Empty state ── */}
        {!hasTimeline && !hasTranscript && !hasFrames && (
          <p className="vp-empty">
            Video analysis data will appear here after processing.
          </p>
        )}
      </div>

      <p className="vp-filename">{fileName ?? "No file"}</p>
    </section>
  );
}
