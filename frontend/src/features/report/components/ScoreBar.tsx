interface ScoreBarProps {
  label: string;
  value: number;
  maxValue?: number;
  color?: string;
  showPercent?: boolean;
}

export function ScoreBar(props: ScoreBarProps): JSX.Element {
  const { label, value, maxValue = 1, color, showPercent = true } = props;
  const pct = Math.max(0, Math.min(100, (value / maxValue) * 100));

  return (
    <div className="score-bar-row">
      <div className="score-bar-label-row">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value">
          {showPercent ? `${Math.round(pct)}%` : value.toFixed(2)}
        </span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{
            width: `${pct}%`,
            background: color ?? `linear-gradient(90deg, rgba(168,130,255,0.9), rgba(120,90,220,0.85))`
          }}
        />
      </div>
    </div>
  );
}

interface ScoreComponentsChartProps {
  components: {
    semantic_relevance: number;
    intent_alignment: number;
    performance_quality: number;
    reference_usefulness: number;
    support_confidence: number;
  };
  compact?: boolean;
}

const COMPONENT_COLORS: Record<string, string> = {
  semantic_relevance: "linear-gradient(90deg, #a78bfa, #7c3aed)",
  intent_alignment: "linear-gradient(90deg, #67e8f9, #06b6d4)",
  reference_usefulness: "linear-gradient(90deg, #fbbf24, #f59e0b)",
  support_confidence: "linear-gradient(90deg, #f9a8d4, #ec4899)"
};

const COMPONENT_LABELS: Record<string, string> = {
  semantic_relevance: "Semantic Relevance",
  intent_alignment: "Intent Alignment",
  reference_usefulness: "Reference Usefulness",
  support_confidence: "Support Confidence"
};

// Keys to display — excludes performance_quality (always 0)
const DISPLAY_KEYS = [
  "semantic_relevance",
  "intent_alignment",
  "reference_usefulness",
  "support_confidence"
];

export function ScoreComponentsChart(props: ScoreComponentsChartProps): JSX.Element {
  const { components, compact = false } = props;

  return (
    <div className={`score-components-chart ${compact ? "score-components-compact" : ""}`}>
      {DISPLAY_KEYS.map((key) => (
        <ScoreBar
          key={key}
          label={COMPONENT_LABELS[key] ?? key}
          value={(components as Record<string, number>)[key] ?? 0}
          maxValue={1}
          color={COMPONENT_COLORS[key]}
          showPercent
        />
      ))}
    </div>
  );
}
