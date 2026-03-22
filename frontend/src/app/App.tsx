import { useMemo } from "react";
import { AppShell } from "./AppShell";
import { UploadPage } from "../features/upload/UploadPage";
import { ApiChatService } from "../services/api/chatApi";
import { MockVideoAnalysisService } from "../services/mock/mockUploadApi";

export function App(): JSX.Element {
  const analysisService = useMemo(() => new MockVideoAnalysisService(), []);
  const chatService = useMemo(() => new ApiChatService(), []);

  return (
    <AppShell>
      <UploadPage analysisService={analysisService} chatService={chatService} />
    </AppShell>
  );
}
