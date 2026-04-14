import { CollapseIcon, ExpandIcon, ExportIcon } from "./ReportIcons";
import type { ReportHeaderData, ReportMeta } from "../types";

interface ReportHeaderProps {
  header: ReportHeaderData;
  meta: ReportMeta;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onShowPreview: () => void;
  onExport: () => void;
}

function getSourceLabel(source: string): { label: string; color: string } {
  switch (source) {
    case "python-service":
      return { label: "Live Model", color: "rgba(34,197,94,0.85)" };
    case "fallback-bundle":
      return { label: "Fallback", color: "rgba(251,191,36,0.85)" };
    default:
      return { label: "Local", color: "rgba(148,163,184,0.7)" };
  }
}

export function ReportHeader(props: ReportHeaderProps): JSX.Element {
  const { header, meta, isExpanded, onToggleExpand, onShowPreview, onExport } = props;
  const source = getSourceLabel(meta.recommender_source);

  return (
    <header className="report-header">
      <div className="report-header-main">
        <h2>{header.title}</h2>
        <p>{header.subtitle}</p>
      </div>

      <div className="report-meta-row">
        <span className="meta-pill" style={{ borderColor: source.color, color: source.color }}>
          {source.label}
        </span>
        <span className="meta-pill">k={header.badges.candidates_k}</span>
        <span className="meta-pill">{header.badges.model}</span>
        <span className="meta-pill">{meta.evidence_label}</span>
        <span className="meta-pill">{meta.confidence_label}</span>
        {meta.fallback_mode && (
          <span className="meta-pill meta-pill-warn">Fallback active</span>
        )}
      </div>

      <div className="report-header-actions">
        {isExpanded ? (
          <button
            type="button"
            className="report-ghost-action"
            onClick={onShowPreview}
          >
            Show preview
          </button>
        ) : null}

        <button type="button" className="report-action-button" onClick={onExport}>
          <ExportIcon className="report-action-icon" />
          <span>Export</span>
        </button>

        <button
          type="button"
          className="report-action-button"
          onClick={onToggleExpand}
          aria-label={isExpanded ? "Collapse report" : "Expand report"}
          title={isExpanded ? "Collapse" : "Expand"}
        >
          {isExpanded ? (
            <CollapseIcon className="report-action-icon" />
          ) : (
            <ExpandIcon className="report-action-icon" />
          )}
          <span>{isExpanded ? "Collapse" : "Expand"}</span>
        </button>
      </div>
    </header>
  );
}
