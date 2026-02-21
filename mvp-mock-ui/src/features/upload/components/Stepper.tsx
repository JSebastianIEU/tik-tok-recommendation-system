import type {
  ProcessingStatus,
  ProcessingStep
} from "../hooks/useProcessingFlow";

type StepIconId = "upload" | "analyze" | "extract" | "compare" | "report" | "done";

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
      <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
        <path d="M12 3V15M12 3L8 7M12 3L16 7M5 16V19H19V16" />
      </svg>
    );
  }

  if (iconId === "analyze") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
        <circle cx="11" cy="11" r="6" />
        <path d="M20 20L16.4 16.4" />
      </svg>
    );
  }

  if (iconId === "extract") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
        <path d="M5 5H11V11H5Z" />
        <path d="M13 5H19V11H13Z" />
        <path d="M5 13H11V19H5Z" />
        <path d="M13 13H19V19H13Z" />
      </svg>
    );
  }

  if (iconId === "compare") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
        <path d="M4 7H14" />
        <path d="M10 17H20" />
        <circle cx="16" cy="7" r="2" />
        <circle cx="8" cy="17" r="2" />
      </svg>
    );
  }

  if (iconId === "report") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
        <path d="M7 3H14L18 7V21H7Z" />
        <path d="M14 3V7H18" />
        <path d="M10 12H15" />
        <path d="M10 16H15" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="processing-inline-icon">
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
    if (index < currentStepIndex) {
      return "done";
    }
    if (index === currentStepIndex) {
      return "error";
    }
    return "future";
  }

  if (status === "done") {
    if (index <= lastIndex) {
      return "done";
    }
    return "future";
  }

  if (index < currentStepIndex) {
    return "done";
  }

  if (index === currentStepIndex) {
    return "current";
  }

  return "future";
}

export function Stepper(props: StepperProps): JSX.Element {
  const { steps, currentStepIndex, status } = props;
  const lastStepIndex = steps.length - 1;

  return (
    <ol className="processing-stepper">
      {steps.map((step, index) => {
        const stepState = getStepState(
          index,
          currentStepIndex,
          status,
          lastStepIndex
        );

        return (
          <li
            className={`processing-step processing-step-${stepState}`}
            key={step.id}
          >
            <span className="processing-step-status" aria-hidden="true">
              {stepState === "done" ? (
                "✓"
              ) : stepState === "current" ? (
                <span className="processing-step-spinner" />
              ) : stepState === "error" ? (
                "!"
              ) : (
                "•"
              )}
            </span>
            <span className="processing-step-icon" aria-hidden="true">
              <StepIcon iconId={step.icon as StepIconId} />
            </span>
            <span className="processing-step-label">{step.label}</span>
          </li>
        );
      })}
    </ol>
  );
}
