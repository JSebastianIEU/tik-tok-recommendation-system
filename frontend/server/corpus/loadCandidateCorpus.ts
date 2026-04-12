import fs from "node:fs/promises";
import path from "node:path";
import { parseDemoDatasetJsonl } from "../../src/services/data/parseDemoDatasetJsonl";
import type { DemoVideoAuthor, DemoVideoRecord } from "../../src/services/data/types";
import {
  RECOMMENDER_CORPUS_BUNDLE_PATH,
  RECOMMENDER_CORPUS_PROVIDER
} from "../config";

const DEMO_DATASET_FILE = path.resolve(process.cwd(), "src/data/demodata.jsonl");
const MAX_COMMENTS_PER_VIDEO = 8;

export type CandidateCorpusProvider = "artifact_bundle" | "demo";

export interface CandidateCorpus {
  provider: CandidateCorpusProvider;
  retrievalMode: "python_bundle" | "supplied_candidates";
  records: DemoVideoRecord[];
  byVideoId: Map<string, DemoVideoRecord>;
}

interface ContractBundleAuthor {
  author_id?: unknown;
  username?: unknown;
  followers_count?: unknown;
}

interface ContractBundleVideo {
  video_id?: unknown;
  author_id?: unknown;
  caption?: unknown;
  hashtags?: unknown;
  keywords?: unknown;
  search_query?: unknown;
  posted_at?: unknown;
  video_url?: unknown;
  duration_seconds?: unknown;
  language?: unknown;
}

interface ContractBundleVideoSnapshot {
  video_id?: unknown;
  event_time?: unknown;
  scraped_at?: unknown;
  ingested_at?: unknown;
  views?: unknown;
  likes?: unknown;
  comments_count?: unknown;
  shares?: unknown;
}

interface ContractBundleComment {
  video_id?: unknown;
  text?: unknown;
  created_at?: unknown;
}

interface ContractBundle {
  authors?: ContractBundleAuthor[];
  videos?: ContractBundleVideo[];
  video_snapshots?: ContractBundleVideoSnapshot[];
  comments?: ContractBundleComment[];
}

interface CachedCorpusRecord {
  corpus: CandidateCorpus;
  mtimeMs: number;
  sourcePath: string;
}

let cachedArtifactCorpus: CachedCorpusRecord | null = null;

function toStringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function toStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function toNumberValue(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function toTimestamp(value: unknown): number {
  const text = toStringValue(value);
  if (!text) {
    return 0;
  }
  const parsed = Date.parse(text);
  return Number.isFinite(parsed) ? parsed : 0;
}

function buildLookup(records: DemoVideoRecord[]): Map<string, DemoVideoRecord> {
  return new Map(records.map((record) => [record.video_id, record]));
}

export function buildArtifactBundleRecords(bundle: ContractBundle): DemoVideoRecord[] {
  const authorsById = new Map<string, ContractBundleAuthor>();
  for (const author of Array.isArray(bundle.authors) ? bundle.authors : []) {
    const authorId = toStringValue(author.author_id);
    if (authorId) {
      authorsById.set(authorId, author);
    }
  }

  const latestSnapshotByVideoId = new Map<string, ContractBundleVideoSnapshot>();
  for (const snapshot of Array.isArray(bundle.video_snapshots) ? bundle.video_snapshots : []) {
    const videoId = toStringValue(snapshot.video_id);
    if (!videoId) {
      continue;
    }
    const existing = latestSnapshotByVideoId.get(videoId);
    const candidateTimestamp = Math.max(
      toTimestamp(snapshot.event_time),
      toTimestamp(snapshot.scraped_at),
      toTimestamp(snapshot.ingested_at)
    );
    const existingTimestamp = existing
      ? Math.max(
          toTimestamp(existing.event_time),
          toTimestamp(existing.scraped_at),
          toTimestamp(existing.ingested_at)
        )
      : 0;
    if (!existing || candidateTimestamp >= existingTimestamp) {
      latestSnapshotByVideoId.set(videoId, snapshot);
    }
  }

  const commentsByVideoId = new Map<
    string,
    Array<{ text: string; createdAt: number }>
  >();
  for (const comment of Array.isArray(bundle.comments) ? bundle.comments : []) {
    const videoId = toStringValue(comment.video_id);
    const text = toStringValue(comment.text);
    if (!videoId || !text) {
      continue;
    }
    const bucket = commentsByVideoId.get(videoId) ?? [];
    bucket.push({
      text,
      createdAt: toTimestamp(comment.created_at)
    });
    commentsByVideoId.set(videoId, bucket);
  }

  const records: DemoVideoRecord[] = [];
  for (const video of Array.isArray(bundle.videos) ? bundle.videos : []) {
    const videoId = toStringValue(video.video_id);
    if (!videoId) {
      continue;
    }
    const authorId = toStringValue(video.author_id);
    const authorRow = authorId ? authorsById.get(authorId) : undefined;
    const snapshot = latestSnapshotByVideoId.get(videoId);
    const groupedComments = commentsByVideoId.get(videoId) ?? [];
    groupedComments.sort((left, right) => right.createdAt - left.createdAt);

    const author: DemoVideoAuthor = {
      ...(authorId ? { author_id: authorId } : {}),
      ...(toStringValue(authorRow?.username)
        ? { username: toStringValue(authorRow?.username) }
        : {}),
      ...(toNumberValue(authorRow?.followers_count) > 0
        ? { followers: toNumberValue(authorRow?.followers_count) }
        : {})
    };

    records.push({
      video_id: videoId,
      caption: toStringValue(video.caption),
      hashtags: toStringArray(video.hashtags),
      keywords: toStringArray(video.keywords),
      metrics: {
        views: toNumberValue(snapshot?.views),
        likes: toNumberValue(snapshot?.likes),
        comments_count: toNumberValue(snapshot?.comments_count),
        shares: toNumberValue(snapshot?.shares)
      },
      author,
      comments: groupedComments
        .slice(0, MAX_COMMENTS_PER_VIDEO)
        .map((entry) => entry.text),
      search_query: toStringValue(video.search_query) || undefined,
      posted_at: toStringValue(video.posted_at) || undefined,
      video_url: toStringValue(video.video_url) || undefined,
      language: toStringValue(video.language) || undefined,
      video_meta: {
        duration_seconds: toNumberValue(video.duration_seconds),
        language: toStringValue(video.language) || undefined
      }
    });
  }

  records.sort((left, right) => {
    const rightPosted = toTimestamp(right.posted_at);
    const leftPosted = toTimestamp(left.posted_at);
    if (rightPosted !== leftPosted) {
      return rightPosted - leftPosted;
    }
    return right.metrics.views - left.metrics.views;
  });
  return records;
}

async function loadDemoCorpus(): Promise<CandidateCorpus> {
  const raw = await fs.readFile(DEMO_DATASET_FILE, "utf-8");
  const records = parseDemoDatasetJsonl(raw);
  return {
    provider: "demo",
    retrievalMode: "supplied_candidates",
    records,
    byVideoId: buildLookup(records)
  };
}

async function loadArtifactBundleCorpus(bundlePath: string): Promise<CandidateCorpus> {
  const resolvedPath = path.resolve(process.cwd(), bundlePath);
  const stats = await fs.stat(resolvedPath);
  if (
    cachedArtifactCorpus &&
    cachedArtifactCorpus.sourcePath === resolvedPath &&
    cachedArtifactCorpus.mtimeMs === stats.mtimeMs
  ) {
    return cachedArtifactCorpus.corpus;
  }

  const raw = await fs.readFile(resolvedPath, "utf-8");
  const bundle = JSON.parse(raw) as ContractBundle;
  const records = buildArtifactBundleRecords(bundle);
  const corpus: CandidateCorpus = {
    provider: "artifact_bundle",
    retrievalMode: "python_bundle",
    records,
    byVideoId: buildLookup(records)
  };
  cachedArtifactCorpus = {
    corpus,
    mtimeMs: stats.mtimeMs,
    sourcePath: resolvedPath
  };
  return corpus;
}

export async function loadCandidateCorpus(): Promise<CandidateCorpus> {
  if (RECOMMENDER_CORPUS_PROVIDER === "demo") {
    return loadDemoCorpus();
  }
  return loadArtifactBundleCorpus(RECOMMENDER_CORPUS_BUNDLE_PATH);
}
