import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface ProcessingStep {
  id: string;
  label: string;
  icon: string;
  durationMs: number;
}

export type ProcessingStatus = "idle" | "processing" | "done" | "error";

/**
 * Progress callback passed to the task so it can advance the stepper
 * in real-time as each phase completes.
 */
export type ProgressCallback = (stepId: string) => void;

interface StartProcessingParams<TResult> {
  task: (onProgress: ProgressCallback) => Promise<TResult>;
  onSuccess: (result: TResult) => void;
  onError?: (error: unknown) => void;
}

interface UseProcessingFlowParams {
  steps: ProcessingStep[];
}

interface UseProcessingFlowResult {
  status: ProcessingStatus;
  currentStepIndex: number;
  errorMessage: string | null;
  totalDurationMs: number;
  startProcessing: <TResult>(params: StartProcessingParams<TResult>) => void;
  resetProcessing: () => void;
}

const PROCESSING_ERROR_MESSAGE =
  "A processing error happened. Please try again.";

export function useProcessingFlow(
  params: UseProcessingFlowParams
): UseProcessingFlowResult {
  const { steps } = params;

  const [status, setStatus] = useState<ProcessingStatus>("idle");
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const flowIdRef = useRef<number>(0);
  const timersRef = useRef<number[]>([]);

  const totalDurationMs = useMemo(() => {
    return steps.reduce((total, step) => total + step.durationMs, 0);
  }, [steps]);

  const stepIdToIndex = useMemo(() => {
    const map = new Map<string, number>();
    steps.forEach((step, idx) => map.set(step.id, idx));
    return map;
  }, [steps]);

  const clearTimers = useCallback((): void => {
    for (const timerId of timersRef.current) {
      window.clearTimeout(timerId);
    }
    timersRef.current = [];
  }, []);

  const resetProcessing = useCallback((): void => {
    flowIdRef.current += 1;
    clearTimers();
    setStatus("idle");
    setCurrentStepIndex(-1);
    setErrorMessage(null);
  }, [clearTimers]);

  const startProcessing = useCallback(
    <TResult,>(paramsToStart: StartProcessingParams<TResult>): void => {
      const { task, onSuccess, onError } = paramsToStart;

      flowIdRef.current += 1;
      const currentFlowId = flowIdRef.current;
      let settled = false;

      const failProcessing = (error: unknown): void => {
        if (settled || flowIdRef.current !== currentFlowId) {
          return;
        }
        settled = true;
        clearTimers();
        setStatus("error");
        setErrorMessage(PROCESSING_ERROR_MESSAGE);
        onError?.(error);
      };

      clearTimers();
      setStatus("processing");
      setErrorMessage(null);
      setCurrentStepIndex(steps.length > 0 ? 0 : -1);

      // Progress callback: task calls onProgress("stepId") to advance
      const onProgress: ProgressCallback = (stepId: string) => {
        if (flowIdRef.current !== currentFlowId || settled) return;
        const idx = stepIdToIndex.get(stepId);
        if (idx !== undefined) {
          setCurrentStepIndex(idx);
        }
      };

      // Also schedule timer-based fallback progression so if the task
      // doesn't call onProgress, steps still advance visually.
      let accumulatedMs = steps.length > 0 ? steps[0].durationMs : 0;
      for (let index = 1; index < steps.length; index += 1) {
        const capturedIndex = index;
        const timerId = window.setTimeout(() => {
          if (flowIdRef.current !== currentFlowId) return;
          // Only advance if we haven't already passed this step
          setCurrentStepIndex((prev) => Math.max(prev, capturedIndex));
        }, accumulatedMs);
        timersRef.current.push(timerId);
        accumulatedMs += steps[index].durationMs;
      }

      const taskPromise = task(onProgress);

      taskPromise
        .then((result) => {
          if (settled || flowIdRef.current !== currentFlowId) return;
          settled = true;
          clearTimers();
          setCurrentStepIndex(steps.length > 0 ? steps.length - 1 : -1);
          setStatus("done");
          onSuccess(result);
        })
        .catch(failProcessing);
    },
    [clearTimers, steps, totalDurationMs, stepIdToIndex]
  );

  useEffect(() => {
    return () => {
      flowIdRef.current += 1;
      clearTimers();
    };
  }, [clearTimers]);

  return {
    status,
    currentStepIndex,
    errorMessage,
    totalDurationMs,
    startProcessing,
    resetProcessing
  };
}
