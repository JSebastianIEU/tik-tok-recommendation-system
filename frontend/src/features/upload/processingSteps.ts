import type { ProcessingStep } from "./hooks/useProcessingFlow";

export const PROCESSING_STEPS: ProcessingStep[] = [
  {
    id: "upload",
    label: "Uploading video...",
    icon: "upload",
    durationMs: 900
  },
  {
    id: "analyze",
    label: "Analyzing content...",
    icon: "analyze",
    durationMs: 2400
  },
  {
    id: "extract",
    label: "Extracting information...",
    icon: "extract",
    durationMs: 1350
  },
  {
    id: "compare",
    label: "Comparing results...",
    icon: "compare",
    durationMs: 1450
  },
  {
    id: "report",
    label: "Building report...",
    icon: "report",
    durationMs: 2650
  },
  {
    id: "done",
    label: "Done",
    icon: "done",
    durationMs: 140
  }
];
