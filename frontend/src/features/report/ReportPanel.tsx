import { useMemo, useRef, useState } from "react";
import type { ReportOutput } from "./types";
import { ComparableDetailsDrawer } from "./components/ComparableDetailsDrawer";
import { ComparablesSection } from "./components/ComparablesSection";
import { DirectComparisonSection } from "./components/DirectComparisonSection";
import { ExecutiveSummarySection } from "./components/ExecutiveSummarySection";
import { RecommendationsSection } from "./components/RecommendationsSection";
import { RelevantCommentsSection } from "./components/RelevantCommentsSection";
import { ReportHeader } from "./components/ReportHeader";
import { exportReportHtml } from "./utils/exportReportHtml";

interface ReportPanelProps {
  report: ReportOutput;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onShowPreview: () => void;
}

export function ReportPanel(props: ReportPanelProps): JSX.Element {
  const { report, isExpanded, onToggleExpand, onShowPreview } = props;
  const reportRef = useRef<HTMLElement>(null);
  const [selectedComparableId, setSelectedComparableId] = useState<string | null>(
    null
  );

  const selectedComparable = useMemo(() => {
    if (!selectedComparableId) {
      return null;
    }

    return (
      report.comparables.find((comparable) => comparable.id === selectedComparableId) ??
      null
    );
  }, [report.comparables, selectedComparableId]);

  return (
    <section ref={reportRef} className="glass-card report-panel">
      <div className="report-scroll-container">
        <ReportHeader
          header={report.header}
          isExpanded={isExpanded}
          onToggleExpand={onToggleExpand}
          onShowPreview={onShowPreview}
          onExport={() => exportReportHtml(reportRef.current)}
        />

        <div className="report-content">
          <p className="report-baseline-disclaimer">{report.header.disclaimer}</p>

          <ExecutiveSummarySection summary={report.executive_summary} />

          <section className="report-section">
            <div className="report-section-head">
              <h3>Analysis reading</h3>
            </div>
            <p className="report-analysis-text">
              {report.executive_summary.summary_text}
            </p>
          </section>

          <ComparablesSection
            items={report.comparables}
            selectedComparableId={selectedComparableId}
            onSelectComparable={setSelectedComparableId}
          />

          <DirectComparisonSection comparison={report.direct_comparison} />

          <RelevantCommentsSection section={report.relevant_comments} />

          <RecommendationsSection section={report.recommendations} />
        </div>
      </div>

      <ComparableDetailsDrawer
        item={selectedComparable}
        onClose={() => setSelectedComparableId(null)}
      />
    </section>
  );
}
