import type { ExplainabilityReportSection, ReportReasoning } from "../types";
import { ScoreComponentsChart } from "./ScoreBar";

interface ExplainabilitySectionProps {
  section?: ExplainabilityReportSection;
  reasoning: ReportReasoning;
  onViewExplainability: () => void;
  onViewCounterfactual: (candidateId: string) => void;
}

function formatConfidence(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

export function ExplainabilitySection(
  props: ExplainabilitySectionProps
): JSX.Element | null {
  const { section, reasoning } = props;

  if (!section) {
    return null;
  }

  // Aggregate score components from top candidates
  const topCandidates = reasoning.evidence_pack.top_candidates.slice(0, 5);
  const avgComponents = reasoning.evidence_pack.aggregate_patterns.score_component_averages;
  const branchMix = reasoning.evidence_pack.candidate_summary.branch_mix;
  const supportMix = reasoning.evidence_pack.candidate_summary.support_mix;

  // Filter counterfactuals that actually have data (not all "unknown")
  const usefulCounterfactuals = section.counterfactual_actions.filter((action) =>
    action.scenarios.some((s) => s.feasibility !== "unknown")
  );

  return (
    <section className="report-section" aria-labelledby="explainability-title">
      <div className="report-section-head">
        <h3 id="explainability-title">Model Explainability</h3>
        <p>How the recommendation engine scored and selected comparables</p>
      </div>

      {/* Average score components across all candidates */}
      {avgComponents && (
        <div className="explain-avg-section">
          <h4>Average Score Components</h4>
          <p className="explain-avg-subtitle">
            Mean scores across top {topCandidates.length} candidates
          </p>
          <ScoreComponentsChart components={avgComponents} />
        </div>
      )}

      {/* Retrieval branch distribution */}
      {Object.keys(branchMix).length > 0 && (
        <div className="explain-branch-section">
          <h4>Retrieval Branches</h4>
          <p className="explain-avg-subtitle">
            How candidates were retrieved from the corpus
          </p>
          <div className="explain-branch-chips">
            {Object.entries(branchMix)
              .sort(([, a], [, b]) => b - a)
              .map(([branch, count]) => (
                <span className="explain-branch-chip" key={branch}>
                  <span className="explain-branch-name">{branch.replace(/_/g, " ")}</span>
                  <span className="explain-branch-count">{count}</span>
                </span>
              ))}
          </div>
        </div>
      )}

      {/* Support distribution */}
      <div className="explain-support-section">
        <h4>Support Distribution</h4>
        <div className="explain-support-bar">
          {supportMix.full > 0 && (
            <div
              className="explain-support-seg explain-support-full"
              style={{ flex: supportMix.full }}
              title={`Full support: ${supportMix.full}`}
            >
              {supportMix.full} full
            </div>
          )}
          {supportMix.partial > 0 && (
            <div
              className="explain-support-seg explain-support-partial"
              style={{ flex: supportMix.partial }}
              title={`Partial support: ${supportMix.partial}`}
            >
              {supportMix.partial} partial
            </div>
          )}
          {supportMix.low > 0 && (
            <div
              className="explain-support-seg explain-support-low"
              style={{ flex: supportMix.low }}
              title={`Low support: ${supportMix.low}`}
            >
              {supportMix.low} low
            </div>
          )}
        </div>
      </div>

      {/* Repeated patterns */}
      {reasoning.evidence_pack.aggregate_patterns.repeated_hashtags.length > 0 && (
        <div className="explain-patterns-section">
          <h4>Common Hashtags Across Comparables</h4>
          <div className="explain-pattern-chips">
            {reasoning.evidence_pack.aggregate_patterns.repeated_hashtags
              .slice(0, 10)
              .map((item) => (
                <span className="explain-pattern-chip" key={item.tag}>
                  <span>#{item.tag.replace(/^#/, "")}</span>
                  <span className="explain-pattern-count">{item.support_count}x</span>
                </span>
              ))}
          </div>
        </div>
      )}

      {/* Counterfactuals — only if they have real data */}
      {usefulCounterfactuals.length > 0 && (
        <div className="explain-cf-section">
          <h4>What-If Scenarios</h4>
          <div className="explain-cf-list">
            {usefulCounterfactuals.slice(0, 3).map((action) => (
              <article
                className="explain-cf-card"
                key={`${action.candidate_id}-${action.rank}`}
              >
                <div className="explain-cf-header">
                  <span className="explain-ev-rank">#{action.rank}</span>
                </div>
                <div className="explain-cf-scenarios">
                  {action.scenarios
                    .filter((s) => s.feasibility !== "unknown")
                    .slice(0, 3)
                    .map((s) => {
                      const delta = s.expected_rank_delta_band as Record<string, number>;
                      return (
                        <div className="explain-cf-scenario" key={s.scenario_id}>
                          <span className="explain-cf-name">
                            {s.scenario_id.replace(/_/g, " ")}
                          </span>
                          <span className={`explain-cf-feasibility explain-cf-${s.feasibility}`}>
                            {s.feasibility}
                          </span>
                          {delta?.p50 != null && (
                            <span className="explain-cf-delta">
                              {delta.p50 > 0 ? "+" : ""}{delta.p50} ranks
                            </span>
                          )}
                          {s.reason && <p className="explain-cf-reason">{s.reason}</p>}
                        </div>
                      );
                    })}
                </div>
              </article>
            ))}
          </div>
        </div>
      )}

      {/* Evidence quality footer */}
      <div className="explain-quality-bar">
        <span>Evidence:</span>
        <span className={reasoning.reasoning_metadata.evidence_sufficiency ? "explain-quality-ok" : "explain-quality-low"}>
          {reasoning.reasoning_metadata.evidence_sufficiency ? "Sufficient" : "Limited"}
        </span>
        <span className="explain-quality-sep">|</span>
        <span>Confidence: {formatConfidence(reasoning.reasoning_metadata.reasoning_confidence)}</span>
        {reasoning.reasoning_metadata.missing_evidence_flags.length > 0 && (
          <>
            <span className="explain-quality-sep">|</span>
            <span className="explain-quality-flags">
              Gaps: {reasoning.reasoning_metadata.missing_evidence_flags.join(", ")}
            </span>
          </>
        )}
      </div>

      <p className="report-section-note">{section.disclaimer}</p>
    </section>
  );
}
