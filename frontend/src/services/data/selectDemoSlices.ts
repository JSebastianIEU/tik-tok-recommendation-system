import type { DemoVideoRecord } from "./types";

function startsWithAny(value: string, prefixes: string[]): boolean {
  const normalized = value.toLowerCase();
  return prefixes.some((prefix) => normalized.startsWith(prefix.toLowerCase()));
}

export function getSeedVideo(dataset: DemoVideoRecord[]): DemoVideoRecord | null {
  return dataset.find((record) => record.video_id === "s001") ?? null;
}

export function getKeywordCandidates(dataset: DemoVideoRecord[]): DemoVideoRecord[] {
  return dataset.filter((record) => startsWithAny(record.video_id, ["c", "b"]));
}

export function getHashtagCandidates(dataset: DemoVideoRecord[]): DemoVideoRecord[] {
  return dataset.filter((record) => startsWithAny(record.video_id, ["h", "i"]));
}
