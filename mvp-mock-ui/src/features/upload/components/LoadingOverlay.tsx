interface LoadingOverlayProps {
  label: string;
}

export function LoadingOverlay(props: LoadingOverlayProps): JSX.Element {
  const { label } = props;

  return (
    <div className="loading-overlay" aria-live="polite" aria-label={label}>
      <div className="spinner" aria-hidden="true" />
      <p className="loading-label">{label}</p>
    </div>
  );
}
