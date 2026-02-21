import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface ProcessingStep {
  id: string;
  label: string;
  icon: string;
  durationMs: number;
}

export type ProcessingStatus = "idle" | "processing" | "done" | "error";

interface StartProcessingParams<TResult> {
  task: () => Promise<TResult>;
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

  const timersRef = useRef<number[]>([]);
  const flowIdRef = useRef<number>(0);

  const totalDurationMs = useMemo(() => {
    return steps.reduce((total, step) => total + step.durationMs, 0);
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

      const taskPromise = task();

      // Schedule visual step progression independently from business logic.
      let accumulatedMs = steps.length > 0 ? steps[0].durationMs : 0;
      for (let index = 1; index < steps.length; index += 1) {
        const timerId = window.setTimeout(() => {
          if (flowIdRef.current !== currentFlowId) {
            return;
          }
          setCurrentStepIndex(index);
        }, accumulatedMs);

        timersRef.current.push(timerId);
        accumulatedMs += steps[index].durationMs;
      }

      const finishTimerId = window.setTimeout(async () => {
        if (flowIdRef.current !== currentFlowId) {
          return;
        }

        try {
          const result = await taskPromise;
          if (settled || flowIdRef.current !== currentFlowId) {
            return;
          }

          settled = true;
          setCurrentStepIndex(steps.length > 0 ? steps.length - 1 : -1);
          setStatus("done");
          onSuccess(result);
        } catch (error) {
          failProcessing(error);
        }
      }, totalDurationMs);

      timersRef.current.push(finishTimerId);

      void taskPromise.catch(failProcessing);
    },
    [clearTimers, steps, totalDurationMs]
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
