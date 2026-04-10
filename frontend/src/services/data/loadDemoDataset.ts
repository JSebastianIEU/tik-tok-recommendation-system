import raw from "../../data/demodata.jsonl?raw";
import { parseDemoDatasetJsonl } from "./parseDemoDatasetJsonl";
import type { DemoVideoRecord } from "./types";

export function loadDemoDataset(): DemoVideoRecord[] {
  return parseDemoDatasetJsonl(raw);
}
