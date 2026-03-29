export type ReportPolarity = "Positive" | "Negative" | "Question";
export type ReportPriority = "High" | "Medium" | "Low";
export type ReportEffort = "Low" | "Medium" | "High";

export interface ReportHeaderBadge {
  candidates_k: number;
  model: string;
  mode: string;
}

export interface ReportHeaderData {
  title: string;
  subtitle: string;
  badges: ReportHeaderBadge;
  disclaimer: string;
}

export interface ExecutiveMetric {
  id: string;
  label: string;
  value: string;
}

export interface ExecutiveSummaryData {
  metrics: ExecutiveMetric[];
  extracted_keywords: string[];
  meaning_points: string[];
  summary_text: string;
}

export interface ComparableMetrics {
  views: number;
  likes: number;
  comments_count: number;
  shares: number;
  engagement_rate: string;
}

export interface ComparableItem {
  id: string;
  caption: string;
  author: string;
  video_url: string;
  thumbnail_url: string;
  hashtags: string[];
  similarity: number;
  metrics: ComparableMetrics;
  matched_keywords: string[];
  observations: string[];
}

export interface DirectComparisonRow {
  id: string;
  label: string;
  your_value_label: string;
  comparable_value_label: string;
  your_value_pct: number;
  comparable_value_pct: number;
}

export interface DirectComparisonData {
  rows: DirectComparisonRow[];
  note: string;
}

export interface RelevantCommentItem {
  id: string;
  text: string;
  topic: string;
  polarity: ReportPolarity;
  relevance_note: string;
}

export interface RelevantCommentsData {
  items: RelevantCommentItem[];
  disclaimer: string;
}

export interface RecommendationItem {
  id: string;
  title: string;
  priority: ReportPriority;
  effort: ReportEffort;
  evidence: string;
}

export interface RecommendationsData {
  items: RecommendationItem[];
}

export interface ExplainabilityEvidenceCard {
  candidate_id: string;
  rank: number;
  feature_contributions: Record<string, unknown>;
  neighbor_evidence: Record<string, unknown>;
  temporal_confidence_band: Record<string, unknown>;
}

export interface ExplainabilityCounterfactualAction {
  candidate_id: string;
  rank: number;
  scenarios: Array<{
    scenario_id: string;
    expected_rank_delta_band: Record<string, unknown>;
    feasibility: string;
    reason?: string;
    trace?: Record<string, unknown>;
  }>;
}

export interface ExplainabilityReportSection {
  evidence_cards: ExplainabilityEvidenceCard[];
  counterfactual_actions: ExplainabilityCounterfactualAction[];
  disclaimer: string;
  trace_metadata: Record<string, unknown>;
}

export interface ReportOutput {
  header: ReportHeaderData;
  executive_summary: ExecutiveSummaryData;
  comparables: ComparableItem[];
  direct_comparison: DirectComparisonData;
  relevant_comments: RelevantCommentsData;
  recommendations: RecommendationsData;
  explainability?: ExplainabilityReportSection;
}
