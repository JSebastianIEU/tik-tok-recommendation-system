import type {
  ProcessingStatus,
  ProcessingStep
} from "../hooks/useProcessingFlow";

type StepIconId = "upload" | "analyze" | "extract" | "compare" | "report" | "done" | "frames" | "vision" | "transcript" | "timeline";

interface StepperProps {
  steps: ProcessingStep[];
  currentStepIndex: number;
  status: ProcessingStatus;
}

interface StepIconProps {
  iconId: StepIconId;
}

function StepIcon(props: StepIconProps): JSX.Element {
  const { iconId } = props;

  if (iconId === "upload") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M12 3V15M12 3L8 7M12 3L16 7M5 16V19H19V16" />
      </svg>
    );
  }

  if (iconId === "frames") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="M10 4V20M14 4V20" />
      </svg>
    );
  }

  if (iconId === "vision") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <circle cx="12" cy="12" r="3" />
        <path d="M2 12C4 7 8 4 12 4S20 7 22 12C20 17 16 20 12 20S4 17 2 12Z" />
      </svg>
    );
  }

  if (iconId === "transcript") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M12 3C7 3 3 6.5 3 11V21L7 18H12C17 18 21 14.5 21 10S17 3 12 3Z" />
        <path d="M8 9H16M8 13H13" />
      </svg>
    );
  }

  if (iconId === "timeline") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M3 20V10L7 14L11 6L15 12L19 4L21 8" />
        <path d="M3 20H21" />
      </svg>
    );
  }

  if (iconId === "analyze") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <circle cx="11" cy="11" r="6" />
        <path d="M20 20L16.4 16.4" />
      </svg>
    );
  }

  if (iconId === "extract") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M5 5H11V11H5Z" />
        <path d="M13 5H19V11H13Z" />
        <path d="M5 13H11V19H5Z" />
        <path d="M13 13H19V19H13Z" />
      </svg>
    );
  }

  if (iconId === "compare") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M4 7H14" />
        <path d="M10 17H20" />
        <circle cx="16" cy="7" r="2" />
        <circle cx="8" cy="17" r="2" />
      </svg>
    );
  }

  if (iconId === "report") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
        <path d="M7 3H14L18 7V21H7Z" />
        <path d="M14 3V7H18" />
        <path d="M10 12H15" />
        <path d="M10 16H15" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="step-icon-svg">
      <path d="M6 12L10 16L18 8" />
    </svg>
  );
}

function getStepState(
  index: number,
  currentStepIndex: number,
  status: ProcessingStatus,
  lastIndex: number
): "done" | "current" | "future" | "error" {
  if (status === "error") {
    if (index < currentStepIndex) return "done";
    if (index === currentStepIndex) return "error";
    return "future";
  }

  if (status === "done") {
    if (index <= lastIndex) return "done";
    return "future";
  }

  if (index < currentStepIndex) return "done";
  if (index === currentStepIndex) return "current";
  return "future";
}

export function Stepper(props: StepperProps): JSX.Element {
  const { steps, currentStepIndex, status } = props;
  const lastStepIndex = steps.length - 1;

  // Progress percentage
  const totalSteps = steps.length;
  const completedSteps = status === "done" ? totalSteps : currentStepIndex;
  const progressPct = Math.round((completedSteps / totalSteps) * 100);

  return (
    <div className="stepper-container">
      {/* Progress bar */}
      <div className="stepper-progress-track">
        <div
          className="stepper-progress-fill"
          style={{ width: `${progressPct}%` }}
        />
      </div>
      <span className="stepper-progress-label">{progressPct}%</span>

      {/* Steps */}
      <ol className="stepper-list">
        {steps.map((step, index) => {
          const stepState = getStepState(index, currentStepIndex, status, lastStepIndex);

          return (
            <li className={`step step-${stepState}`} key={step.id}>
              <span className="step-indicator">
                {stepState === "done" ? (
                  <svg viewBox="0 0 24 24" className="step-check-svg">
                    <path d="M6 12L10 16L18 8" />
                  </svg>
                ) : stepState === "current" ? (
                  <span className="step-pulse" />
                ) : stepState === "error" ? (
                  <span className="step-error-x">!</span>
                ) : (
                  <span className="step-dot" />
                )}
              </span>
              <span className="step-icon">
                <StepIcon iconId={step.icon as StepIconId} />
              </span>
              <span className="step-label">{step.label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
