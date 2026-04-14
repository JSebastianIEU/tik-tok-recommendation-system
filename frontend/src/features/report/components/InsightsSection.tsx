import type { ReportReasoning } from "../types";
import { ScoreBar } from "./ScoreBar";

interface InsightsSectionProps {
  reasoning: ReportReasoning;
}

const MATCH_BARS: Array<{
  key: keyof ReportReasoning["evidence_pack"]["aggregate_patterns"]["score_component_averages"];
  label: string;
  color: string;
}> = [
  { key: "semantic_relevance", label: "Topic Match", color: "linear-gradient(90deg, #a78bfa, #7c3aed)" },
  { key: "intent_alignment", label: "Goal Alignment", color: "linear-gradient(90deg, #67e8f9, #06b6d4)" },
  { key: "reference_usefulness", label: "Content Quality", color: "linear-gradient(90deg, #fbbf24, #f59e0b)" },
  { key: "support_confidence", label: "Model Confidence", color: "linear-gradient(90deg, #f9a8d4, #ec4899)" }
];

export function InsightsSection(props: InsightsSectionProps): JSX.Element | null {
  const { reasoning } = props;
  const avg = reasoning.evidence_pack.aggregate_patterns.score_component_averages;
  const repeatedHashtags = reasoning.evidence_pack.aggregate_patterns.repeated_hashtags;
  const confidence = reasoning.reasoning_metadata.reasoning_confidence;

  const hasAvg = avg && Object.values(avg).some((v) => v > 0);
  const hasHashtags = repeatedHashtags.length > 0;

  if (!hasAvg && !hasHashtags) {
    return null;
  }

  return (
    <section className="report-section" aria-labelledby="insights-title">
      <div className="report-section-head">
        <h3 id="insights-title">Insights</h3>
        <p>Patterns across your comparable videos</p>
      </div>

      {hasAvg && (
        <div className="insights-match-section">
          <h4>How Your Video Matches</h4>
          <p className="insights-subtitle">
            Average similarity scores across the top comparables
          </p>
          <div className="insights-bars">
            {MATCH_BARS.map((bar) => (
              <ScoreBar
                key={bar.key}
                label={bar.label}
                value={avg[bar.key] ?? 0}
                maxValue={1}
                color={bar.color}
                showPercent
              />
            ))}
          </div>
        </div>
      )}

      {hasHashtags && (
        <div className="insights-common-section">
          <h4>What Similar Videos Have in Common</h4>
          <p className="insights-subtitle">
            Hashtags that appear most frequently in your top matches
          </p>
          <div className="insights-hashtag-list">
            {repeatedHashtags.slice(0, 10).map((item) => (
              <span className="insights-hashtag-chip" key={item.tag}>
                <span className="insights-hashtag-name">
                  #{item.tag.replace(/^#/, "")}
                </span>
                <span className="insights-hashtag-count">
                  {item.support_count}x
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="insights-confidence-note">
        {confidence >= 0.6
          ? "These results have moderate-to-high confidence based on available data."
          : "These results are directional — limited data means lower confidence."}
      </div>
    </section>
  );
}
