interface MetricCardProps {
  label: string;
  value: string | number;
  icon?: string;
  subtitle?: string;
  accentColor?: string;
}

export function MetricCard(props: MetricCardProps): JSX.Element {
  const { label, value, icon, subtitle, accentColor } = props;

  return (
    <article
      className="metric-card"
      style={accentColor ? { borderColor: accentColor } : undefined}
    >
      {icon ? <span className="metric-card-icon">{icon}</span> : null}
      <div className="metric-card-body">
        <p className="metric-card-value">{value}</p>
        <p className="metric-card-label">{label}</p>
        {subtitle ? <p className="metric-card-subtitle">{subtitle}</p> : null}
      </div>
    </article>
  );
}

interface StatRowProps {
  items: Array<{ label: string; value: string | number; icon?: string }>;
}

export function StatRow(props: StatRowProps): JSX.Element {
  return (
    <div className="stat-row">
      {props.items.map((item) => (
        <div className="stat-row-item" key={item.label}>
          {item.icon ? <span className="stat-row-icon">{item.icon}</span> : null}
          <span className="stat-row-value">{item.value}</span>
          <span className="stat-row-label">{item.label}</span>
        </div>
      ))}
    </div>
  );
}
