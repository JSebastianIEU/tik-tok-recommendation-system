export type ReportPolarity = "Positive" | "Negative" | "Question";
export type ReportPriority = "High" | "Medium" | "Low";
export type ReportEffort = "Low" | "Medium" | "High";
export type ReportEvidenceLabel = "Strong evidence" | "Moderate evidence" | "Limited evidence";
export type ReportConfidenceLabel = "High confidence" | "Medium confidence" | "Low confidence";
export type SupportLevel = "full" | "partial" | "low" | "unknown";

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

export interface ReportMeta {
  request_id: string;
  objective: string;
  objective_effective: string;
  generated_at: string;
  recommender_source: "python-service" | "fallback-bundle" | "deterministic-local";
  fallback_mode: boolean;
  fallback_reason?: string | null;
  evidence_label: ReportEvidenceLabel;
  confidence_label: ReportConfidenceLabel;
  experiment_id?: string | null;
  variant?: "control" | "treatment" | null;
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
  candidate_id: string;
  caption: string;
  author: string;
  video_url: string;
  thumbnail_url: string;
  hashtags: string[];
  similarity: number;
  support_level: SupportLevel;
  confidence_label: ReportConfidenceLabel;
  metrics: ComparableMetrics;
  matched_keywords: string[];
  observations: string[];
  why_this_was_chosen: string;
  ranking_reasons: string[];
  score_components: {
    semantic_relevance: number;
    intent_alignment: number;
    performance_quality: number;
    reference_usefulness: number;
    support_confidence: number;
  };
  retrieval_branches: string[];
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
  rationale: string;
  confidence_label: ReportConfidenceLabel;
  effect_area:
    | "hook"
    | "clarity"
    | "cta"
    | "pacing"
    | "format"
    | "audience_alignment"
    | "topic_alignment";
  caveats: string[];
  evidence_refs: string[];
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

export interface ReasoningEvidencePack {
  version: string;
  request: {
    request_id: string;
    objective: string;
    objective_effective: string;
    fallback_mode: boolean;
  };
  query_summary: {
    description: string;
    hashtags: string[];
    mentions: string[];
    content_type?: string;
    primary_cta?: string;
    audience?: {
      label?: string;
      segments?: string[];
      expertise_level?: string;
    };
    locale?: string;
    language?: string;
  };
  candidate_summary: {
    final_count: number;
    top_k_considered: number;
    support_mix: {
      full: number;
      partial: number;
      low: number;
    };
    branch_mix: Record<string, number>;
  };
  top_candidates: Array<{
    candidate_id: string;
    rank: number;
    score: number;
    support_level: SupportLevel;
    score_components: ComparableItem["score_components"];
    ranking_reasons: string[];
    hashtags?: string[];
    content_type?: string;
    language?: string;
    locale?: string;
  }>;
  aggregate_patterns: {
    repeated_hashtags: Array<{ tag: string; support_count: number }>;
    repeated_content_types: Array<{ content_type: string; support_count: number }>;
    repeated_ranking_reasons: Array<{ reason: string; support_count: number }>;
    score_component_averages: ComparableItem["score_components"];
  };
  contrast_signals: {
    top_vs_rest: Array<{
      feature: string;
      direction: "higher" | "lower" | "mixed";
      magnitude: number;
    }>;
    mismatches: string[];
    conflicts: string[];
  };
  evidence_quality: {
    sufficient: boolean;
    confidence: number;
    missing_flags: string[];
  };
}

export interface ExplanationUnit {
  explanation_id: string;
  claim_type:
    | "selection_reason"
    | "pattern_summary"
    | "draft_gap"
    | "strength"
    | "risk"
    | "counterfactual_hint";
  statement: string;
  evidence_refs: string[];
  confidence: number;
  status: "strong" | "moderate" | "weak" | "fallback";
  caveats: string[];
}

export interface RecommendationUnit {
  recommendation_id: string;
  action: string;
  rationale: string;
  priority: ReportPriority;
  effort: ReportEffort;
  confidence: number;
  evidence_refs: string[];
  expected_effect_area: RecommendationItem["effect_area"];
  caveats: string[];
}

export interface ReasoningMetadata {
  version: string;
  fallback_mode: boolean;
  evidence_sufficiency: boolean;
  reasoning_confidence: number;
  missing_evidence_flags: string[];
}

export interface ReportReasoning {
  evidence_pack: ReasoningEvidencePack;
  explanation_units: ExplanationUnit[];
  recommendation_units: RecommendationUnit[];
  reasoning_metadata: ReasoningMetadata;
}

export interface ReportOutput {
  meta: ReportMeta;
  header: ReportHeaderData;
  executive_summary: ExecutiveSummaryData;
  comparables: ComparableItem[];
  direct_comparison: DirectComparisonData;
  relevant_comments: RelevantCommentsData;
  recommendations: RecommendationsData;
  reasoning: ReportReasoning;
  explainability?: ExplainabilityReportSection;
}
