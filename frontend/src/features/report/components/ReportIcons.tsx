interface IconProps {
  className?: string;
}

export function ExpandIcon(props: IconProps): JSX.Element {
  const { className } = props;
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path d="M8 3H3V8" />
      <path d="M16 3H21V8" />
      <path d="M8 21H3V16" />
      <path d="M16 21H21V16" />
      <path d="M3 8L9 2" />
      <path d="M21 8L15 2" />
      <path d="M3 16L9 22" />
      <path d="M21 16L15 22" />
    </svg>
  );
}

export function CollapseIcon(props: IconProps): JSX.Element {
  const { className } = props;
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path d="M9 9H3V3" />
      <path d="M15 9H21V3" />
      <path d="M9 15H3V21" />
      <path d="M15 15H21V21" />
      <path d="M3 3L9 9" />
      <path d="M21 3L15 9" />
      <path d="M3 21L9 15" />
      <path d="M21 21L15 15" />
    </svg>
  );
}

export function ExportIcon(props: IconProps): JSX.Element {
  const { className } = props;
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path d="M12 3V14" />
      <path d="M8.5 6.5L12 3L15.5 6.5" />
      <path d="M4 14V20H20V14" />
    </svg>
  );
}

export function CloseIcon(props: IconProps): JSX.Element {
  const { className } = props;
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path d="M6 6L18 18" />
      <path d="M18 6L6 18" />
    </svg>
  );
}
