import { useEffect, useMemo, useState } from "react";
import type { ReportOutput } from "../../report/types";
import { generateReport } from "../../../services/api/reportApi";
import type { IVideoAnalysisService } from "../../../services/contracts/IVideoAnalysisService";
import type {
  SignalHintsPayload,
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
  isAnalyzing: boolean;
  loadingLabel: string;
  processingSteps: ProcessingStep[];
  processingStatus: ProcessingStatus;
  currentStepIndex: number;
  processingErrorMessage: string | null;
  setDescription: (value: string) => void;
  setMentions: (values: string[]) => void;
  setHashtags: (values: string[]) => void;
  setObjective: (value: UploadFormValues["objective"]) => void;
  setAudience: (value: string) => void;
  setContentType: (value: UploadFormValues["content_type"]) => void;
  setPrimaryCta: (value: UploadFormValues["primary_cta"]) => void;
  setLocale: (value: string) => void;
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
  description: "",
  objective: "engagement",
  audience: "",
  content_type: "showcase",
  primary_cta: "none",
  locale:
    typeof navigator !== "undefined" && navigator.language
      ? navigator.language.toLowerCase()
      : "en"
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

function buildSignalHints(
  formValues: UploadFormValues,
  analysisResult?: VideoAnalysisResult | null
): SignalHintsPayload {
  if (analysisResult?.signal_hints) {
    return {
      ...analysisResult.signal_hints,
      transcript_text: analysisResult.transcript || formValues.description,
      video_caption: analysisResult.video_caption || "",
      ocr_text: analysisResult.ocr_text || ""
    };
  }
  return {
    transcript_text: formValues.description.trim() || undefined
  };
}

function mergeSignalHints(
  ...sources: Array<SignalHintsPayload | undefined>
): SignalHintsPayload | undefined {
  const merged: SignalHintsPayload = {};

  for (const source of sources) {
    if (!source) {
      continue;
    }
    for (const [key, value] of Object.entries(source)) {
      if (value === undefined || value === null) {
        continue;
      }
      if (typeof value === "string" && !value.trim()) {
        continue;
      }
      const typedKey = key as keyof SignalHintsPayload;
      merged[typedKey] = value;
    }
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
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
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [preAnalysis, setPreAnalysis] = useState<VideoAnalysisResult | null>(null);

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

  const setObjective = (value: UploadFormValues["objective"]): void => {
    setFormValues((previous) => ({
      ...previous,
      objective: value
    }));
  };

  const setAudience = (value: string): void => {
    setFormValues((previous) => ({
      ...previous,
      audience: value
    }));
  };

  const setContentType = (value: UploadFormValues["content_type"]): void => {
    setFormValues((previous) => ({
      ...previous,
      content_type: value
    }));
  };

  const setPrimaryCta = (value: UploadFormValues["primary_cta"]): void => {
    setFormValues((previous) => ({
      ...previous,
      primary_cta: value
    }));
  };

  const setLocale = (value: string): void => {
    setFormValues((previous) => ({
      ...previous,
      locale: value
    }));
  };

  const onFileSelected = (file: File | null): void => {
    setSelectedFile(file);
    setPreAnalysis(null);
    setIsAnalyzing(false);

    if (!file) {
      return;
    }

    setError(null);
    setAnalysisResult(null);
    setReportResult(null);
    if (processingFlow.status !== "processing") {
      setPhase("idle");
    }

    if (!file.type.startsWith("video/")) {
      return;
    }

    // Fire background video analysis immediately on file select
    setIsAnalyzing(true);
    const capturedFile = file;

    analysisService
      .analyzeVideo({
        file: capturedFile,
        mentions: [],
        hashtags: [],
        description: "",
        objective: "engagement",
        content_type: "showcase",
        primary_cta: "none",
        locale: formValues.locale
      })
      .then((result) => {
        // Only apply if this file is still the selected one
        setSelectedFile((current) => {
          if (current !== capturedFile) {
            return current;
          }

          setPreAnalysis(result);
          setIsAnalyzing(false);

          // Pre-fill description from VLM caption or transcript
          const suggestedDescription =
            result.video_caption || result.transcript || "";
          if (suggestedDescription) {
            setFormValues((prev) => ({
              ...prev,
              description: prev.description || suggestedDescription
            }));
          }

          // Pre-fill hashtags from keywords
          const suggestedHashtags = result.keyTopics
            .map((kw) => kw.replace(/\s+/g, "").toLowerCase())
            .filter((tag) => tag.length > 1)
            .slice(0, 5);
          if (suggestedHashtags.length > 0) {
            setFormValues((prev) => ({
              ...prev,
              hashtags:
                prev.hashtags.length === 0 ? suggestedHashtags : prev.hashtags
            }));
          }

          return current;
        });
      })
      .catch((err) => {
        console.warn("Background video analysis failed:", err);
        setIsAnalyzing(false);
      });
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
        const requestSignalHints = buildSignalHints(formValues);

        // Reuse pre-analysis from file-select if available; otherwise analyze now
        const analysis = preAnalysis ?? await analysisService.analyzeVideo({
          file: selectedFile,
          mentions: formValues.mentions,
          hashtags: formValues.hashtags,
          description: formValues.description,
          objective: formValues.objective,
          audience: formValues.audience,
          content_type: formValues.content_type,
          primary_cta: formValues.primary_cta,
          locale: formValues.locale,
          signal_hints: requestSignalHints
        });
        const mergedSignalHints = mergeSignalHints(
          analysis.signal_hints,
          requestSignalHints
        );
        const reportSignalHints = mergeSignalHints(
          mergedSignalHints,
          buildSignalHints(formValues, analysis)
        );

        const elapsedMs = Date.now() - taskStartedAt;
        const remainingBeforeReportStep = Math.max(
          0,
          REPORT_STEP_START_DELAY_MS - elapsedMs
        );

        if (remainingBeforeReportStep > 0) {
          await wait(remainingBeforeReportStep);
        }

        const report = await generateReport({
          asset_id: analysis.asset_id,
          mentions: formValues.mentions,
          hashtags: formValues.hashtags,
          description: formValues.description,
          objective: formValues.objective,
          audience: formValues.audience,
          content_type: formValues.content_type,
          primary_cta: formValues.primary_cta,
          locale: formValues.locale,
          signal_hints: reportSignalHints
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
    isAnalyzing,
    loadingLabel,
    processingSteps: PROCESSING_STEPS,
    processingStatus: processingFlow.status,
    currentStepIndex: processingFlow.currentStepIndex,
    processingErrorMessage: processingFlow.errorMessage,
    setDescription,
    setMentions,
    setHashtags,
    setObjective,
    setAudience,
    setContentType,
    setPrimaryCta,
    setLocale,
    onFileSelected,
    retryProcessing,
    submit
  };
}
