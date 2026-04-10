import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type LabelingReviewLabel = "saved" | "relevant" | "not_relevant";

const REVIEW_LABELS = new Set<LabelingReviewLabel>([
  "saved",
  "relevant",
  "not_relevant"
]);

const SESSION_VERSION = "labeling.review_session.v1";
const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../.."
);

interface RawBenchmarkCandidate {
  candidate_id?: unknown;
  display?: unknown;
  candidate_payload?: unknown;
  baseline_rank?: unknown;
  baseline_score?: unknown;
  support_level?: unknown;
  ranking_reasons?: unknown;
}

interface RawBenchmarkCase {
  case_id?: unknown;
  objective?: unknown;
  query?: unknown;
  candidates?: unknown;
  retrieve_k?: unknown;
  label_pool_size?: unknown;
  source_candidate_pool_size?: unknown;
  notes?: unknown;
}

interface RawBenchmarkDataset {
  version?: unknown;
  generated_at?: unknown;
  bundle_dir?: unknown;
  sample_metadata?: unknown;
  rubric?: unknown;
  cases?: unknown;
}

export interface LabelingSourceSummary {
  source_id: string;
  file_name: string;
  source_path: string;
  generated_at: string;
  case_count: number;
  objectives: string[];
}

export interface LabelingCandidateReview {
  label: LabelingReviewLabel | null;
  note: string;
  updated_at: string | null;
}

export interface LabelingSessionCandidate {
  candidate_id: string;
  display: Record<string, unknown>;
  candidate_payload: Record<string, unknown>;
  baseline_rank: number | null;
  baseline_score: number | null;
  support_level: string | null;
  ranking_reasons: string[];
  review: LabelingCandidateReview;
}

export interface LabelingSessionCase {
  case_id: string;
  objective: string;
  query: {
    query_id: string;
    display: Record<string, unknown>;
    query_payload: Record<string, unknown>;
  };
  retrieve_k: number;
  label_pool_size: number;
  source_candidate_pool_size: number;
  notes: string;
  candidates: LabelingSessionCandidate[];
}

export interface LabelingSessionSummary {
  case_count: number;
  candidate_count: number;
  reviewed_count: number;
  saved_count: number;
  relevant_count: number;
  not_relevant_count: number;
  completion_ratio: number;
}

export interface LabelingSession {
  version: string;
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  storage_path: string;
  source: LabelingSourceSummary;
  rubric: {
    version: string;
    labels: LabelingReviewLabel[];
    instructions: string[];
  };
  cases: LabelingSessionCase[];
  summary: LabelingSessionSummary;
}

export interface LabelingSessionListItem {
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  storage_path: string;
  source: LabelingSourceSummary;
  summary: LabelingSessionSummary;
}

interface PersistedLabelingSession {
  version: string;
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  source: LabelingSourceSummary;
  rubric: {
    version: string;
    labels: LabelingReviewLabel[];
    instructions: string[];
  };
  cases: LabelingSessionCase[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asInt(value: unknown, fallback = 0): number {
  const parsed = asNumberOrNull(value);
  return parsed === null ? fallback : Math.max(0, Math.round(parsed));
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter(Boolean);
}

function nowIso(): string {
  return new Date().toISOString();
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function reviewSummary(cases: LabelingSessionCase[]): LabelingSessionSummary {
  let candidateCount = 0;
  let reviewedCount = 0;
  let savedCount = 0;
  let relevantCount = 0;
  let notRelevantCount = 0;

  for (const itemCase of cases) {
    candidateCount += itemCase.candidates.length;
    for (const candidate of itemCase.candidates) {
      const label = candidate.review.label;
      if (!label) {
        continue;
      }
      reviewedCount += 1;
      if (label === "saved") {
        savedCount += 1;
      } else if (label === "relevant") {
        relevantCount += 1;
      } else if (label === "not_relevant") {
        notRelevantCount += 1;
      }
    }
  }

  return {
    case_count: cases.length,
    candidate_count: candidateCount,
    reviewed_count: reviewedCount,
    saved_count: savedCount,
    relevant_count: relevantCount,
    not_relevant_count: notRelevantCount,
    completion_ratio:
      candidateCount > 0 ? Number((reviewedCount / candidateCount).toFixed(4)) : 0
  };
}

function hydrateSession(
  raw: PersistedLabelingSession,
  storagePath: string
): LabelingSession {
  return {
    ...raw,
    storage_path: storagePath,
    summary: reviewSummary(raw.cases)
  };
}

function normalizeBenchmarkSource(
  sourcePath: string,
  payload: RawBenchmarkDataset
): LabelingSourceSummary {
  const cases = Array.isArray(payload.cases) ? payload.cases : [];
  const objectives = Array.from(
    new Set(
      cases
        .map((item) => (isRecord(item) ? asString(item.objective) : ""))
        .filter(Boolean)
    )
  ).sort();

  return {
    source_id: path.basename(sourcePath),
    file_name: path.basename(sourcePath),
    source_path: sourcePath,
    generated_at: asString(payload.generated_at, ""),
    case_count: cases.length,
    objectives
  };
}

function buildSessionCase(rawCase: RawBenchmarkCase): LabelingSessionCase {
  const query = isRecord(rawCase.query) ? rawCase.query : {};
  const rawCandidates = Array.isArray(rawCase.candidates) ? rawCase.candidates : [];

  return {
    case_id: asString(rawCase.case_id),
    objective: asString(rawCase.objective),
    query: {
      query_id: asString(query.query_id),
      display: isRecord(query.display) ? query.display : {},
      query_payload: isRecord(query.query_payload) ? query.query_payload : {}
    },
    retrieve_k: asInt(rawCase.retrieve_k),
    label_pool_size: asInt(rawCase.label_pool_size),
    source_candidate_pool_size: asInt(rawCase.source_candidate_pool_size),
    notes: asString(rawCase.notes),
    candidates: rawCandidates
      .map((rawCandidate): LabelingSessionCandidate | null => {
        const candidate = isRecord(rawCandidate) ? rawCandidate as RawBenchmarkCandidate : null;
        if (!candidate) {
          return null;
        }
        const candidateId = asString(candidate.candidate_id);
        if (!candidateId) {
          return null;
        }
        return {
          candidate_id: candidateId,
          display: isRecord(candidate.display) ? candidate.display : {},
          candidate_payload: isRecord(candidate.candidate_payload)
            ? candidate.candidate_payload
            : {},
          baseline_rank: asNumberOrNull(candidate.baseline_rank),
          baseline_score: asNumberOrNull(candidate.baseline_score),
          support_level: (() => {
            const value = asString(candidate.support_level);
            return value || null;
          })(),
          ranking_reasons: asStringArray(candidate.ranking_reasons),
          review: {
            label: null,
            note: "",
            updated_at: null
          }
        };
      })
      .filter((item): item is LabelingSessionCandidate => item !== null)
  };
}

async function readJsonFile<T>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

async function writeJsonFile(filePath: string, payload: unknown): Promise<void> {
  const next = `${filePath}.tmp`;
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(next, JSON.stringify(payload, null, 2), "utf-8");
  await fs.rename(next, filePath);
}

function defaultSessionName(sourceId: string): string {
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  return `${slugify(sourceId.replace(/\.json$/i, "")) || "labeling"}-${stamp}`;
}

export class LabelingSessionStore {
  private readonly sessionsDir: string;
  private readonly benchmarksDir: string;

  constructor(options?: { sessionsDir?: string; benchmarksDir?: string }) {
    this.sessionsDir =
      options?.sessionsDir ??
      path.resolve(REPO_ROOT, "artifacts/labeling_sessions");
    this.benchmarksDir =
      options?.benchmarksDir ??
      path.resolve(REPO_ROOT, "artifacts/benchmarks");
  }

  private sessionPath(sessionId: string): string {
    return path.join(this.sessionsDir, `${sessionId}.json`);
  }

  async listSources(): Promise<LabelingSourceSummary[]> {
    await fs.mkdir(this.benchmarksDir, { recursive: true });
    const entries = await fs.readdir(this.benchmarksDir, { withFileTypes: true });
    const sources: LabelingSourceSummary[] = [];

    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) {
        continue;
      }
      if (entry.name.includes("_eval_")) {
        continue;
      }
      const filePath = path.join(this.benchmarksDir, entry.name);
      try {
        const payload = await readJsonFile<RawBenchmarkDataset>(filePath);
        if (!Array.isArray(payload.cases)) {
          continue;
        }
        sources.push(normalizeBenchmarkSource(filePath, payload));
      } catch {
        continue;
      }
    }

    sources.sort((left, right) => left.file_name.localeCompare(right.file_name));
    return sources;
  }

  preferredSource(sources: LabelingSourceSummary[]): LabelingSourceSummary | null {
    if (sources.length === 0) {
      return null;
    }
    return (
      sources.find((item) => item.file_name.includes("training")) ??
      sources.find((item) => item.file_name.includes("seed")) ??
      sources[0] ??
      null
    );
  }

  async listSessions(): Promise<LabelingSessionListItem[]> {
    await fs.mkdir(this.sessionsDir, { recursive: true });
    const entries = await fs.readdir(this.sessionsDir, { withFileTypes: true });
    const sessions: LabelingSessionListItem[] = [];

    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) {
        continue;
      }
      const filePath = path.join(this.sessionsDir, entry.name);
      try {
        const payload = await readJsonFile<PersistedLabelingSession>(filePath);
        const hydrated = hydrateSession(payload, filePath);
        sessions.push({
          session_id: hydrated.session_id,
          session_name: hydrated.session_name,
          created_at: hydrated.created_at,
          updated_at: hydrated.updated_at,
          storage_path: hydrated.storage_path,
          source: hydrated.source,
          summary: hydrated.summary
        });
      } catch {
        continue;
      }
    }

    sessions.sort((left, right) => right.updated_at.localeCompare(left.updated_at));
    return sessions;
  }

  async createSession(options?: {
    sourceId?: string;
    sessionName?: string;
  }): Promise<LabelingSession> {
    const sources = await this.listSources();
    const selectedSource =
      sources.find((item) => item.source_id === options?.sourceId) ??
      this.preferredSource(sources);
    if (!selectedSource) {
      throw new Error("no_labeling_sources_available");
    }

    const sourcePayload = await readJsonFile<RawBenchmarkDataset>(
      selectedSource.source_path
    );
    const sessionId =
      `${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}-${slugify(selectedSource.file_name) || "session"}`;
    const createdAt = nowIso();
    const payload: PersistedLabelingSession = {
      version: SESSION_VERSION,
      session_id: sessionId,
      session_name:
        (options?.sessionName?.trim() || defaultSessionName(selectedSource.file_name)).slice(
          0,
          96
        ),
      created_at: createdAt,
      updated_at: createdAt,
      source: selectedSource,
      rubric: {
        version: "labeling.saved_relevant_not_relevant.v1",
        labels: ["saved", "relevant", "not_relevant"],
        instructions: [
          "Use saved for the strongest examples you would actively keep as a reference.",
          "Use relevant for candidates that are meaningfully comparable but not top-tier saves.",
          "Use not relevant for poor or misleading comparables."
        ]
      },
      cases: (Array.isArray(sourcePayload.cases) ? sourcePayload.cases : [])
        .map((item) => (isRecord(item) ? buildSessionCase(item as RawBenchmarkCase) : null))
        .filter((item): item is LabelingSessionCase => item !== null)
    };

    const filePath = this.sessionPath(sessionId);
    await writeJsonFile(filePath, payload);
    return hydrateSession(payload, filePath);
  }

  async loadSession(sessionId: string): Promise<LabelingSession> {
    const filePath = this.sessionPath(sessionId);
    const payload = await readJsonFile<PersistedLabelingSession>(filePath);
    return hydrateSession(payload, filePath);
  }

  async updateCandidateReview(input: {
    sessionId: string;
    caseId: string;
    candidateId: string;
    label: LabelingReviewLabel | null;
    note?: string;
  }): Promise<LabelingSession> {
    const filePath = this.sessionPath(input.sessionId);
    const payload = await readJsonFile<PersistedLabelingSession>(filePath);
    const caseRecord = payload.cases.find((item) => item.case_id === input.caseId);
    if (!caseRecord) {
      throw new Error("labeling_case_not_found");
    }
    const candidate = caseRecord.candidates.find(
      (item) => item.candidate_id === input.candidateId
    );
    if (!candidate) {
      throw new Error("labeling_candidate_not_found");
    }

    if (input.label !== null && !REVIEW_LABELS.has(input.label)) {
      throw new Error("labeling_label_invalid");
    }

    candidate.review = {
      label: input.label,
      note: typeof input.note === "string" ? input.note.slice(0, 500) : candidate.review.note,
      updated_at: nowIso()
    };
    payload.updated_at = nowIso();

    await writeJsonFile(filePath, payload);
    return hydrateSession(payload, filePath);
  }
}
