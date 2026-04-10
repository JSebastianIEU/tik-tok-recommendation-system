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

export function ReportHeader(props: ReportHeaderProps): JSX.Element {
  const { header, meta, isExpanded, onToggleExpand, onShowPreview, onExport } = props;

  return (
    <header className="report-header">
      <div className="report-header-main">
        <h2>{header.title}</h2>
        <p>{header.subtitle}</p>
      </div>

      <div className="report-meta-chips">
        <span className="report-chip">Candidates: k={header.badges.candidates_k}</span>
        <span className="report-chip">Model: {header.badges.model}</span>
        <span className="report-chip">Mode: {header.badges.mode}</span>
        <span className="report-chip">{meta.evidence_label}</span>
        <span className="report-chip">{meta.confidence_label}</span>
        {meta.fallback_mode ? <span className="report-chip">Fallback active</span> : null}
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
