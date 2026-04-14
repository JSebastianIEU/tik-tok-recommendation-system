import { useEffect, useMemo, useRef, useState } from "react";
import { sendReportFeedback } from "../../services/api/reportFeedbackApi";
import type { ComparableItem, ReportOutput } from "./types";
import { ComparableDetailsDrawer } from "./components/ComparableDetailsDrawer";
import { ComparablesSection } from "./components/ComparablesSection";
import { ExecutiveSummarySection } from "./components/ExecutiveSummarySection";
import { InsightsSection } from "./components/InsightsSection";
import { ReportHeader } from "./components/ReportHeader";
import { exportReportHtml } from "./utils/exportReportHtml";

export interface HashtagSuggestionItem {
  hashtag: string;
  score: number;
  frequency: number;
  avg_engagement: number;
}

interface ReportPanelProps {
  report: ReportOutput;
  suggestedHashtags?: HashtagSuggestionItem[];
  userHashtags?: string[];
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
  const { report, suggestedHashtags = [], userHashtags = [], isExpanded, onToggleExpand, onShowPreview } = props;
  const reportRef = useRef<HTMLElement>(null);
  const viewedRef = useRef<string | null>(null);
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

  const trackComparableOpen = (item: ComparableItem): void => {
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name: "comparable_opened",
      entity_type: "comparable",
      entity_id: item.candidate_id,
      section: "comparables",
      rank: Number(item.id.split("-").at(-1) ?? 0) || undefined,
      signal_strength: "medium",
      label_direction: "positive",
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
      void sendReportFeedback({
        ...buildFeedbackBase(report),
        event_name: "comparable_details_opened",
        entity_type: "comparable",
        entity_id: selected.candidate_id,
        section: "comparables",
        signal_strength: "medium",
        label_direction: "neutral"
      });
    }
  };

  const [feedbackState, setFeedbackState] = useState<
    Record<string, "relevant" | "not_relevant" | "saved" | undefined>
  >({});

  const handleMarkRelevant = (item: ComparableItem, label: "relevant" | "not_relevant"): void => {
    setFeedbackState((prev) => ({ ...prev, [item.candidate_id]: label }));
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name: label === "relevant" ? "comparable_marked_relevant" : "comparable_marked_not_relevant",
      entity_type: "comparable",
      entity_id: item.candidate_id,
      section: "comparables",
      signal_strength: "strong",
      label_direction: label === "relevant" ? "positive" : "negative",
      metadata: { hashtags: item.hashtags, author: item.author }
    });
  };

  const handleSaveComparable = (item: ComparableItem): void => {
    setFeedbackState((prev) => ({ ...prev, [item.candidate_id]: "saved" }));
    void sendReportFeedback({
      ...buildFeedbackBase(report),
      event_name: "comparable_saved",
      entity_type: "comparable",
      entity_id: item.candidate_id,
      section: "comparables",
      signal_strength: "strong",
      label_direction: "positive",
      metadata: { hashtags: item.hashtags, author: item.author }
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
          <ExecutiveSummarySection
            summary={report.executive_summary}
            suggestedHashtags={suggestedHashtags}
            userHashtags={userHashtags}
          />

          <ComparablesSection
            items={report.comparables}
            selectedComparableId={selectedComparableId}
            onSelectComparable={handleComparableSelection}
            onOpenVideo={trackComparableOpen}
            onMarkRelevant={handleMarkRelevant}
            onSaveComparable={handleSaveComparable}
            feedbackState={feedbackState}
          />

          <InsightsSection reasoning={report.reasoning} />
        </div>
      </div>

      <ComparableDetailsDrawer
        item={selectedComparable}
        onClose={() => setSelectedComparableId(null)}
      />
    </section>
  );
}
