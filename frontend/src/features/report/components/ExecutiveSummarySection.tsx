import type { ExecutiveSummaryData } from "../types";
import type { HashtagSuggestionItem } from "../ReportPanel";

interface ExecutiveSummarySectionProps {
  summary: ExecutiveSummaryData;
  suggestedHashtags?: HashtagSuggestionItem[];
  userHashtags?: string[];
}

function formatCompact(value: number): string {
  if (value >= 1000) {
    return new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: 1
    }).format(value);
  }
  return value.toLocaleString("en-US");
}

export function ExecutiveSummarySection(
  props: ExecutiveSummarySectionProps
): JSX.Element {
  const { suggestedHashtags = [], userHashtags = [] } = props;

  const hasUserHashtags = userHashtags.length > 0;
  const hasHashtags = suggestedHashtags.length > 0;

  return (
    <section className="report-section" aria-labelledby="report-summary-title">
      <div className="report-section-head">
        <h3 id="report-summary-title">Query Overview</h3>
        <p>Your input hashtags and model-suggested hashtags</p>
      </div>

      {hasUserHashtags && (
        <div className="exec-keywords-section">
          <h4>Your Hashtags</h4>
          <div className="exec-keywords-cloud">
            {userHashtags.map((tag, i) => (
              <span
                className="exec-keyword-pill"
                key={tag}
                style={{ animationDelay: `${i * 60}ms` }}
              >
                #{tag.replace(/^#/, "")}
              </span>
            ))}
          </div>
        </div>
      )}

      {hasHashtags && (
        <div className="exec-hashtags-section">
          <h4>Model-Suggested Hashtags</h4>
          <p className="exec-hashtags-subtitle">
            Ranked by FAISS/SBERT similarity from {formatCompact(26929)} video corpus
          </p>
          <div className="hashtag-table">
            <div className="hashtag-table-header">
              <span className="ht-col-tag">Hashtag</span>
              <span className="ht-col-score">Similarity</span>
              <span className="ht-col-freq">Frequency</span>
              <span className="ht-col-eng">Avg Engagement</span>
            </div>
            {suggestedHashtags.map((item, i) => {
              const scorePct = Math.round(item.score * 100);
              return (
                <div
                  className="hashtag-table-row"
                  key={item.hashtag}
                  style={{ animationDelay: `${i * 40}ms` }}
                >
                  <span className="ht-col-tag">
                    <span className="ht-rank">{i + 1}</span>
                    <span className="ht-name">#{item.hashtag.replace(/^#/, "")}</span>
                  </span>
                  <span className="ht-col-score">
                    <span className="ht-score-bar-track">
                      <span
                        className="ht-score-bar-fill"
                        style={{ width: `${scorePct}%` }}
                      />
                    </span>
                    <span className="ht-score-val">{scorePct}%</span>
                  </span>
                  <span className="ht-col-freq">{formatCompact(item.frequency)}</span>
                  <span className="ht-col-eng">
                    {(item.avg_engagement * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
