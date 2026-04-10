import { useEffect, useRef, useState } from "react";
import type {
  ProcessingStatus,
  ProcessingStep
} from "../hooks/useProcessingFlow";
import { Stepper } from "./Stepper";

interface ProcessingCardProps {
  videoUrl: string | null;
  steps: ProcessingStep[];
  currentStepIndex: number;
  status: ProcessingStatus;
  errorMessage: string | null;
  onRetry: () => void;
}

export function ProcessingCard(props: ProcessingCardProps): JSX.Element {
  const { videoUrl, steps, currentStepIndex, status, errorMessage, onRetry } = props;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewError, setPreviewError] = useState<boolean>(false);

  useEffect(() => {
    if (!videoUrl || !videoRef.current) {
      return;
    }

    const element = videoRef.current;
    const playPromise = element.play();

    if (playPromise) {
      void playPromise.catch(() => {
        setPreviewError(true);
      });
    }
  }, [videoUrl, status]);

  useEffect(() => {
    if (!videoUrl) {
      setPreviewError(false);
    }
  }, [videoUrl]);

  return (
    <section className="glass-card processing-card">
      <div className="processing-preview">
        {videoUrl ? (
          <video
            ref={videoRef}
            className="processing-preview-video"
            src={videoUrl}
            muted
            playsInline
            autoPlay
            loop
            preload="metadata"
            onError={() => setPreviewError(true)}
          />
        ) : (
          <div className="processing-preview-fallback">Preview is not available</div>
        )}

        {previewError ? (
          <div className="processing-preview-fallback">
            The preview video could not be played.
          </div>
        ) : null}
      </div>

      <div className="processing-info">
        <h3 className="panel-title">processing</h3>
        <Stepper
          steps={steps}
          currentStepIndex={currentStepIndex}
          status={status}
        />

        {status === "error" ? (
          <div className="processing-error-box" role="alert">
            <p>{errorMessage ?? "A processing error happened. Try again."}</p>
            <button
              type="button"
              className="processing-retry-button"
              onClick={onRetry}
            >
              Retry
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
