import type { ProcessingStep } from "./hooks/useProcessingFlow";

export const PROCESSING_STEPS: ProcessingStep[] = [
  {
    id: "upload",
    label: "Uploading video...",
    icon: "upload",
    durationMs: 800
  },
  {
    id: "frames",
    label: "Extracting frames & audio...",
    icon: "frames",
    durationMs: 4000
  },
  {
    id: "vision",
    label: "Analyzing scenes & motion...",
    icon: "vision",
    durationMs: 6000
  },
  {
    id: "transcript",
    label: "Transcribing audio...",
    icon: "transcript",
    durationMs: 8000
  },
  {
    id: "timeline",
    label: "Building interactive timeline...",
    icon: "timeline",
    durationMs: 3000
  },
  {
    id: "compare",
    label: "Finding comparable videos...",
    icon: "compare",
    durationMs: 2000
  },
  {
    id: "report",
    label: "Ranking & building report...",
    icon: "report",
    durationMs: 3000
  },
  {
    id: "done",
    label: "Done",
    icon: "done",
    durationMs: 140
  }
];
