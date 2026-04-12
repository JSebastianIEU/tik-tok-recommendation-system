import type { ReactNode } from "react";

export type LayoutMode = "upload" | "split" | "merged" | "results";

interface PanelTransitionWrapperProps {
  mode: LayoutMode;
  className?: string;
  left: ReactNode;
  right: ReactNode;
}

export function PanelTransitionWrapper(
  props: PanelTransitionWrapperProps
): JSX.Element {
  const { mode, className, left, right } = props;

  return (
    <section
      className={`menu-grid panel-transition-wrapper layout-${mode} ${className ?? ""}`.trim()}
    >
      <div className="left-column transition-panel-left">{left}</div>
      <div className="right-column transition-panel-right">{right}</div>
    </section>
  );
}
