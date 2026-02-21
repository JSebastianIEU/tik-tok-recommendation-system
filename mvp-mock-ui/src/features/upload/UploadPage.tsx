import { useEffect, useState } from "react";
import { FloatingChatWidget } from "../chat/components/FloatingChatWidget";
import { ReportPanel } from "../report/ReportPanel";
import type { IChatService } from "../../services/contracts/IChatService";
import type { IVideoAnalysisService } from "../../services/contracts/IVideoAnalysisService";
import {
  PanelTransitionWrapper,
  type LayoutMode
} from "./components/PanelTransitionWrapper";
import { PreviewThumbnailCard } from "./components/PreviewThumbnailCard";
import { ProcessingCard } from "./components/ProcessingCard";
import { UploadCard } from "./components/UploadCard";
import { UploadForm } from "./components/UploadForm";
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
        : "split";

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
      <PreviewThumbnailCard
        videoUrl={uploadWorkflow.previewUrl}
        fileName={uploadWorkflow.selectedFile?.name ?? null}
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
          isExpanded={isReportExpanded}
          onToggleExpand={() => setIsReportExpanded((previous) => !previous)}
          onShowPreview={() => setIsReportExpanded(false)}
        />
      </div>
    ) : (
      <UploadForm
        values={uploadWorkflow.formValues}
        disabled={uploadWorkflow.isBusy}
        error={uploadWorkflow.error}
        onDescriptionChange={uploadWorkflow.setDescription}
        onMentionsChange={uploadWorkflow.setMentions}
        onHashtagsChange={uploadWorkflow.setHashtags}
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
        chatService={chatService}
        resetKey={uploadWorkflow.uploadSession}
      />
    </>
  );
}
