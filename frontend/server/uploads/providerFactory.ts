import { FFPROBE_BIN, UPLOAD_ANALYZER_PROVIDER } from "../config";
import type { AssetAnalysisProvider } from "./contracts";
import { createBaselineAssetAnalysisProvider } from "./providers/baselineAssetAnalyzer";

interface CreateUploadAnalysisProviderOptions {
  provider?: string;
  ffprobeBin?: string;
}

export function createUploadAnalysisProvider(
  options: CreateUploadAnalysisProviderOptions = {}
): AssetAnalysisProvider {
  const provider = (options.provider ?? UPLOAD_ANALYZER_PROVIDER).trim().toLowerCase();

  if (provider === "baseline") {
    return createBaselineAssetAnalysisProvider({
      ffprobeBin: options.ffprobeBin ?? FFPROBE_BIN
    });
  }

  // Future: e.g. `python` / `remote` provider returning AssetAnalysisResult with
  // transcript, timeline, visual_features (see uploads/contracts.ts).

  throw new Error(`Unsupported upload analysis provider: ${provider}`);
}
