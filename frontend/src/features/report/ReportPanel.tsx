import { useEffect, useMemo, useRef, useState } from "react";
import { sendReportFeedback } from "../../services/api/reportFeedbackApi";
import type { ComparableItem, ReportOutput } from "./types";
import { ComparableDetailsDrawer } from "./components/ComparableDetailsDrawer";
import { ComparablesSection } from "./components/ComparablesSection";
import { DirectComparisonSection } from "./components/DirectComparisonSection";
import { ExecutiveSummarySection } from "./components/ExecutiveSummarySection";
import { ExplainabilitySection } from "./components/ExplainabilitySection";
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

function buildFeedbackBase(report: ReportOutput) {
  return {
    request_id: report.meta.request_id,
    objective_effective: report.meta.objective_effective,
    experiment_id: report.meta.experiment_id ?? undefined,
    variant: report.meta.variant ?? undefined
  };
}

export function ReportPanel(props: ReportPanelProps): JSX.Element {
  const { report, isExpanded, onToggleExpand, onShowPreview } = props;
  const reportRef = useRef<HTMLElement>(null);
  const viewedRef = useRef<string | null>(null);
  const [selectedComparableId, setSelectedComparableId] = useState<string | null>(
    null
  );
  const [comparableFeedback, setComparableFeedback] = useState<
    Record<string, "relevant" | "not_relevant" | "saved" | undefined>
  >({});
  const [recommendationFeedback, setRecommendationFeedback] = useState<
    Record<string, "useful" | "not_useful" | "saved" | undefined>
  >({});

  const selectedComparable = useMemo(() => {
    if (!selectedComparableId) {
      return null;
    }

    return (
      report.comparables.find((comparable) => comparable.id === selectedComparableId) ??
      null
    );
  }, [report.comparables, selectedComparableId]);

  useEffect(() => {
    if (!report.meta.request_id || viewedRef.current === report.meta.request_id) {
      return;
    }
    viewedRef.current = report.meta.request_id;
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name: "report_viewed",
      entity_type: "report",
      section: "header",
      signal_strength: "weak",
      label_direction: "context",
      metadata: {
        fallback_mode: report.meta.fallback_mode
      }
    });
  }, [report]);

  const trackComparableFeedback = (
    item: ComparableItem,
    eventName: string,
    signalStrength: "strong" | "medium" | "weak",
    labelDirection: "positive" | "negative" | "neutral" | "context"
  ): void => {
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name: eventName,
      entity_type: "comparable",
      entity_id: item.candidate_id,
      section: "comparables",
      rank: Number(item.id.split("-").at(-1) ?? 0) || undefined,
      signal_strength: signalStrength,
      label_direction: labelDirection,
      metadata: {
        support_level: item.support_level,
        confidence_label: item.confidence_label
      }
    });
  };

  const handleComparableSelection = (id: string): void => {
    setSelectedComparableId(id);
    const selected = report.comparables.find((item) => item.id === id);
    if (selected) {
      trackComparableFeedback(
        selected,
        "comparable_details_opened",
        "medium",
        "neutral"
      );
    }
  };

  const handleComparableLabel = (
    item: ComparableItem,
    label: "relevant" | "not_relevant"
  ): void => {
    setComparableFeedback((previous) => ({ ...previous, [item.id]: label }));
    trackComparableFeedback(
      item,
      label === "relevant"
        ? "comparable_marked_relevant"
        : "comparable_marked_not_relevant",
      "strong",
      label === "relevant" ? "positive" : "negative"
    );
  };

  const handleComparableSave = (item: ComparableItem): void => {
    setComparableFeedback((previous) => ({ ...previous, [item.id]: "saved" }));
    trackComparableFeedback(item, "comparable_saved", "strong", "positive");
  };

  const handleRecommendationFeedback = (
    recommendationId: string,
    label: "useful" | "not_useful" | "saved"
  ): void => {
    setRecommendationFeedback((previous) => ({ ...previous, [recommendationId]: label }));
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name:
        label === "useful"
          ? "recommendation_marked_useful"
          : label === "not_useful"
            ? "recommendation_marked_not_useful"
            : "recommendation_saved",
      entity_type: "recommendation",
      entity_id: recommendationId,
      section: "recommendations",
      signal_strength: label === "saved" ? "strong" : "strong",
      label_direction:
        label === "useful" || label === "saved" ? "positive" : "negative",
      metadata: {}
    });
  };

  return (
    <section ref={reportRef} className="glass-card report-panel">
      <div className="report-scroll-container">
        <ReportHeader
          header={report.header}
          meta={report.meta}
          isExpanded={isExpanded}
          onToggleExpand={onToggleExpand}
          onShowPreview={onShowPreview}
          onExport={() => {
            void sendReportFeedback({
              ...buildFeedbackBase(report),
              event_name: "report_exported",
              entity_type: "report",
              section: "header",
              signal_strength: "medium",
              label_direction: "positive"
            });
            exportReportHtml(reportRef.current);
          }}
        />

        <div className="report-content">
          <p className="report-baseline-disclaimer">{report.header.disclaimer}</p>

          <ExecutiveSummarySection summary={report.executive_summary} />

          <section className="report-section">
            <div className="report-section-head">
              <h3>Analysis reading</h3>
              <p>{report.meta.evidence_label}</p>
            </div>
            <p className="report-analysis-text">
              {report.executive_summary.summary_text}
            </p>
          </section>

          <ComparablesSection
            items={report.comparables}
            selectedComparableId={selectedComparableId}
            onSelectComparable={handleComparableSelection}
            onOpenVideo={(item) => trackComparableFeedback(item, "comparable_opened", "medium", "positive")}
            onMarkRelevant={handleComparableLabel}
            onSaveComparable={handleComparableSave}
            feedbackState={comparableFeedback}
          />

          <DirectComparisonSection comparison={report.direct_comparison} />

          <RelevantCommentsSection section={report.relevant_comments} />

          <RecommendationsSection
            section={report.recommendations}
            onFeedback={handleRecommendationFeedback}
            feedbackState={recommendationFeedback}
          />

          <ExplainabilitySection
            section={report.explainability}
            reasoning={report.reasoning}
            onViewExplainability={() => {
              void sendReportFeedback({
                ...buildFeedbackBase(report),
                event_name: "explainability_viewed",
                entity_type: "explainability",
                section: "explainability",
                signal_strength: "medium",
                label_direction: "neutral"
              });
            }}
            onViewCounterfactual={(candidateId) => {
              void sendReportFeedback({
                ...buildFeedbackBase(report),
                event_name: "counterfactual_viewed",
                entity_type: "explainability",
                entity_id: candidateId,
                section: "explainability",
                signal_strength: "medium",
                label_direction: "neutral"
              });
            }}
          />
        </div>
      </div>

      <ComparableDetailsDrawer
        item={selectedComparable}
        onClose={() => setSelectedComparableId(null)}
      />
    </section>
  );
}
