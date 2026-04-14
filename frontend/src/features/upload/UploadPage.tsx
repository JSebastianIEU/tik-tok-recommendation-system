import { useEffect, useState } from "react";
import { FloatingChatWidget } from "../chat/components/FloatingChatWidget";
import { ReportPanel } from "../report/ReportPanel";
import type { IChatService } from "../../services/contracts/IChatService";
import type { IVideoAnalysisService } from "../../services/contracts/IVideoAnalysisService";
import {
  PanelTransitionWrapper,
  type LayoutMode
} from "./components/PanelTransitionWrapper";
import { ProcessingCard } from "./components/ProcessingCard";
import { UploadCard } from "./components/UploadCard";
import { UploadForm } from "./components/UploadForm";
import { VideoPlayerPanel } from "./components/VideoPlayerPanel";
import { useUploadWorkflow } from "./hooks/useUploadWorkflow";

interface UploadPageProps {
  analysisService: IVideoAnalysisService;
  chatService: IChatService;
}

export function UploadPage(props: UploadPageProps): JSX.Element {
  const { analysisService, chatService } = props;

  const [isReportExpanded, setIsReportExpanded] = useState<boolean>(false);
  const uploadWorkflow = useUploadWorkflow({ analysisService });

  const layoutMode: LayoutMode =
    uploadWorkflow.processingStatus === "processing" ||
    uploadWorkflow.processingStatus === "error"
      ? "merged"
      : uploadWorkflow.phase === "done"
        ? "results"
        : uploadWorkflow.selectedFile
          ? "split"
          : "upload";

  useEffect(() => {
    if (uploadWorkflow.phase !== "done") {
      setIsReportExpanded(false);
    }
  }, [uploadWorkflow.phase]);

  const leftContent =
    layoutMode === "merged" ? (
      <ProcessingCard
        videoUrl={uploadWorkflow.previewUrl}
        steps={uploadWorkflow.processingSteps}
        currentStepIndex={uploadWorkflow.currentStepIndex}
        status={uploadWorkflow.processingStatus}
        errorMessage={uploadWorkflow.processingErrorMessage ?? uploadWorkflow.error}
        onRetry={uploadWorkflow.retryProcessing}
      />
    ) : layoutMode === "results" ? (
      <VideoPlayerPanel
        videoUrl={uploadWorkflow.previewUrl}
        fileName={uploadWorkflow.selectedFile?.name ?? null}
        analysis={uploadWorkflow.analysisResult}
      />
    ) : (
      <UploadCard
        fileName={uploadWorkflow.selectedFile?.name ?? null}
        isBusy={uploadWorkflow.isBusy}
        loadingLabel={uploadWorkflow.loadingLabel}
        onFileSelected={uploadWorkflow.onFileSelected}
      />
    );

  const rightContent =
    layoutMode === "results" && uploadWorkflow.reportResult ? (
      <div className="results-reveal">
        <ReportPanel
          report={uploadWorkflow.reportResult}
          suggestedHashtags={uploadWorkflow.reportHashtags}
          userHashtags={uploadWorkflow.userHashtags}
          isExpanded={isReportExpanded}
          onToggleExpand={() => setIsReportExpanded((previous) => !previous)}
          onShowPreview={() => setIsReportExpanded(false)}
        />
      </div>
    ) : (
      <UploadForm
        values={uploadWorkflow.formValues}
        disabled={uploadWorkflow.isBusy}
        isAnalyzing={uploadWorkflow.isAnalyzing}
        error={uploadWorkflow.error}
        onDescriptionChange={uploadWorkflow.setDescription}
        onMentionsChange={uploadWorkflow.setMentions}
        onHashtagsChange={uploadWorkflow.setHashtags}
        onObjectiveChange={uploadWorkflow.setObjective}
        onAudienceChange={uploadWorkflow.setAudience}
        onContentTypeChange={uploadWorkflow.setContentType}
        onPrimaryCtaChange={uploadWorkflow.setPrimaryCta}
        onLocaleChange={uploadWorkflow.setLocale}
        onSubmit={uploadWorkflow.submit}
      />
    );

  return (
    <>
      <PanelTransitionWrapper
        mode={layoutMode}
        className={isReportExpanded ? "results-expanded" : ""}
        left={leftContent}
        right={rightContent}
      />

      <FloatingChatWidget
        report={uploadWorkflow.reportResult}
        videoAnalysis={uploadWorkflow.analysisResult}
        chatService={chatService}
        resetKey={uploadWorkflow.uploadSession}
      />
    </>
  );
}
