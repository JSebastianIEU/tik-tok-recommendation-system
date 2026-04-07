import { useMemo } from "react";
import { AppShell } from "./AppShell";
import { LabelingPage } from "../features/labeling/LabelingPage";
import { UploadPage } from "../features/upload/UploadPage";
import { ApiChatService } from "../services/api/chatApi";
import { MockVideoAnalysisService } from "../services/mock/mockUploadApi";

export function App(): JSX.Element {
  const analysisService = useMemo(() => new MockVideoAnalysisService(), []);
  const chatService = useMemo(() => new ApiChatService(), []);
  const isLabelingRoute =
    typeof window !== "undefined" &&
    (window.location.pathname.startsWith("/labeling") ||
      window.location.hash === "#labeling");

  return (
    <AppShell>
      {isLabelingRoute ? (
        <LabelingPage />
      ) : (
        <UploadPage analysisService={analysisService} chatService={chatService} />
      )}
    </AppShell>
  );
}
