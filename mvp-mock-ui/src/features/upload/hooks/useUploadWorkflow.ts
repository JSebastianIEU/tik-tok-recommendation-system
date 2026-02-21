import { useEffect, useMemo, useState } from "react";
import type { ReportOutput } from "../../report/types";
import { generateReport } from "../../../services/api/reportApi";
import type { IVideoAnalysisService } from "../../../services/contracts/IVideoAnalysisService";
import type {
  UploadFormValues,
  UploadPhase,
  VideoAnalysisResult
} from "../../../services/contracts/models";
import { PROCESSING_STEPS } from "../processingSteps";
import {
  useProcessingFlow,
  type ProcessingStatus,
  type ProcessingStep
} from "./useProcessingFlow";

interface UseUploadWorkflowParams {
  analysisService: IVideoAnalysisService;
}

interface UseUploadWorkflowResult {
  formValues: UploadFormValues;
  selectedFile: File | null;
  previewUrl: string | null;
  phase: UploadPhase;
  error: string | null;
  analysisResult: VideoAnalysisResult | null;
  reportResult: ReportOutput | null;
  uploadSession: number;
  isBusy: boolean;
  loadingLabel: string;
  processingSteps: ProcessingStep[];
  processingStatus: ProcessingStatus;
  currentStepIndex: number;
  processingErrorMessage: string | null;
  setDescription: (value: string) => void;
  setMentions: (values: string[]) => void;
  setHashtags: (values: string[]) => void;
  onFileSelected: (file: File | null) => void;
  retryProcessing: () => void;
  submit: () => Promise<void>;
}

interface UploadTaskResult {
  analysis: VideoAnalysisResult;
  report: ReportOutput;
}

const INITIAL_FORM_VALUES: UploadFormValues = {
  mentions: [],
  hashtags: [],
  description: ""
};

const PROCESSING_ERROR_MESSAGE =
  "A processing error happened. Please try again.";

const REPORT_STEP_INDEX = PROCESSING_STEPS.findIndex((step) => step.id === "report");
const REPORT_STEP_START_DELAY_MS = PROCESSING_STEPS.slice(
  0,
  Math.max(0, REPORT_STEP_INDEX)
).reduce((total, step) => total + step.durationMs, 0);

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export function useUploadWorkflow(
  params: UseUploadWorkflowParams
): UseUploadWorkflowResult {
  const { analysisService } = params;

  const [formValues, setFormValues] =
    useState<UploadFormValues>(INITIAL_FORM_VALUES);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(
    null
  );
  const [reportResult, setReportResult] = useState<ReportOutput | null>(null);
  const [uploadSession, setUploadSession] = useState<number>(0);

  const processingFlow = useProcessingFlow({ steps: PROCESSING_STEPS });

  const isBusy = phase === "validating" || processingFlow.status === "processing";

  const loadingLabel = useMemo(() => {
    if (
      processingFlow.status === "processing" &&
      processingFlow.currentStepIndex >= 0
    ) {
      return PROCESSING_STEPS[processingFlow.currentStepIndex]?.label ?? "";
    }
    return "";
  }, [processingFlow.currentStepIndex, processingFlow.status]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    // Create and revoke object URLs to avoid memory leaks.
    const nextObjectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextObjectUrl);

    return () => {
      URL.revokeObjectURL(nextObjectUrl);
    };
  }, [selectedFile]);

  const setDescription = (value: string): void => {
    setFormValues((previous) => ({
      ...previous,
      description: value
    }));
  };

  const setMentions = (values: string[]): void => {
    setFormValues((previous) => ({
      ...previous,
      mentions: values
    }));
  };

  const setHashtags = (values: string[]): void => {
    setFormValues((previous) => ({
      ...previous,
      hashtags: values
    }));
  };

  const onFileSelected = (file: File | null): void => {
    setSelectedFile(file);

    if (file) {
      setError(null);
      setAnalysisResult(null);
      setReportResult(null);
      if (processingFlow.status !== "processing") {
        setPhase("idle");
      }
    }
  };

  const retryProcessing = (): void => {
    processingFlow.resetProcessing();
    setPhase("idle");
    setError(null);
    setAnalysisResult(null);
    setReportResult(null);
  };

  const submit = async (): Promise<void> => {
    setPhase("validating");
    setError(null);

    if (!selectedFile) {
      setPhase("error");
      setError("Select a video before uploading.");
      return;
    }

    if (!selectedFile.type.startsWith("video/")) {
      setPhase("error");
      setError("The selected file must be a valid video.");
      return;
    }

    setUploadSession((previous) => previous + 1);
    setAnalysisResult(null);
    setReportResult(null);
    setPhase("processing");

    processingFlow.startProcessing({
      task: async (): Promise<UploadTaskResult> => {
        const taskStartedAt = Date.now();

        const analysis = await analysisService.analyzeVideo({
          file: selectedFile,
          mentions: formValues.mentions,
          hashtags: formValues.hashtags,
          description: formValues.description
        });

        const elapsedMs = Date.now() - taskStartedAt;
        const remainingBeforeReportStep = Math.max(
          0,
          REPORT_STEP_START_DELAY_MS - elapsedMs
        );

        if (remainingBeforeReportStep > 0) {
          await wait(remainingBeforeReportStep);
        }

        const report = await generateReport({
          seed_video_id: "s001",
          mentions: formValues.mentions,
          hashtags: formValues.hashtags,
          description: formValues.description
        });

        return { analysis, report };
      },
      onSuccess: (result) => {
        setAnalysisResult(result.analysis);
        setReportResult(result.report);
        setError(null);
        setPhase("done");
      },
      onError: (processingError) => {
        console.error(processingError);
        setPhase("error");

        if (processingError instanceof Error && processingError.message.trim()) {
          setError(processingError.message);
          return;
        }

        setError(PROCESSING_ERROR_MESSAGE);
      }
    });
  };

  return {
    formValues,
    selectedFile,
    previewUrl,
    phase,
    error,
    analysisResult,
    reportResult,
    uploadSession,
    isBusy,
    loadingLabel,
    processingSteps: PROCESSING_STEPS,
    processingStatus: processingFlow.status,
    currentStepIndex: processingFlow.currentStepIndex,
    processingErrorMessage: processingFlow.errorMessage,
    setDescription,
    setMentions,
    setHashtags,
    onFileSelected,
    retryProcessing,
    submit
  };
}
