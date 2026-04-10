import fs from "node:fs/promises";
import path from "node:path";
import { parseDemoDatasetJsonl } from "../../src/services/data/parseDemoDatasetJsonl";
import type { DemoVideoRecord } from "../../src/services/data/types";

const DATASET_FILE = path.resolve(process.cwd(), "src/data/demodata.jsonl");

export async function loadDatasetFromFile(): Promise<DemoVideoRecord[]> {
  const raw = await fs.readFile(DATASET_FILE, "utf-8");
  return parseDemoDatasetJsonl(raw);
}
