import fs from "node:fs/promises";

export const KNOWLEDGE_BASE_CATEGORIES = [
  "algorithm",
  "hooks",
  "retention",
  "cta",
  "creators",
  "hashtags",
  "content_formats"
] as const;

export const KNOWLEDGE_BASE_CONFIDENCE = ["high", "medium"] as const;
export const KNOWLEDGE_BASE_OBJECTIVES = ["reach", "engagement", "conversion", "community"] as const;

export type KnowledgeBaseCategory = (typeof KNOWLEDGE_BASE_CATEGORIES)[number];
export type KnowledgeBaseConfidence = (typeof KNOWLEDGE_BASE_CONFIDENCE)[number];
export type KnowledgeBaseObjective = (typeof KNOWLEDGE_BASE_OBJECTIVES)[number];

export interface KnowledgeBaseEntry {
  id: string;
  title: string;
  category: KnowledgeBaseCategory;
  content: string[];
  action_hint: string;
  impact_area: string;
  keywords: string[];
  objective_tags?: KnowledgeBaseObjective[];
  updated_at: string;
  confidence: KnowledgeBaseConfidence;
  active: boolean;
}

export interface KnowledgeBaseDataset {
  version: string;
  entries: KnowledgeBaseEntry[];
}

interface ScoredKnowledgeEntry {
  entry: KnowledgeBaseEntry;
  score: number;
  updatedAtEpoch: number;
  lexicalMatches: number;
}

interface PreparedKnowledgeEntry {
  entry: KnowledgeBaseEntry;
  searchTokens: Set<string>;
  keywordPhrases: string[];
  updatedAtEpoch: number;
}

export interface KnowledgeBaseStore {
  version: string;
  entries: PreparedKnowledgeEntry[];
}

export interface KnowledgeBaseSearchParams {
  question: string;
  objective?: string;
  maxResults?: number;
  priority: "high" | "low";
}

export interface KnowledgeBaseSearchResult {
  source: "knowledge_base";
  entries: KnowledgeBaseEntry[];
  matched_categories: KnowledgeBaseCategory[];
  priority: "high" | "low";
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9#\s]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value: string): string[] {
  return normalizeText(value)
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter(Boolean);
}

function assertValidIsoDate(value: string): boolean {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed);
}

function assertAllowed<T extends readonly string[]>(
  value: string,
  allowed: T
): value is T[number] {
  return (allowed as readonly string[]).includes(value);
}

export function validateKnowledgeBaseDataset(dataset: unknown): KnowledgeBaseDataset {
  if (!isObject(dataset)) {
    throw new Error("Knowledge base payload must be an object.");
  }

  const version = typeof dataset.version === "string" ? dataset.version.trim() : "";
  if (!version) {
    throw new Error("Knowledge base version is required.");
  }

  if (!Array.isArray(dataset.entries) || dataset.entries.length === 0) {
    throw new Error("Knowledge base entries must be a non-empty array.");
  }

  const entries: KnowledgeBaseEntry[] = [];
  const ids = new Set<string>();
  let activeCount = 0;

  for (let idx = 0; idx < dataset.entries.length; idx += 1) {
    const raw = dataset.entries[idx];
    if (!isObject(raw)) {
      throw new Error(`Entry at index ${idx} must be an object.`);
    }

    const id = typeof raw.id === "string" ? raw.id.trim() : "";
    if (!id) {
      throw new Error(`Entry at index ${idx} is missing 'id'.`);
    }
    if (ids.has(id)) {
      throw new Error(`Knowledge base entry id '${id}' is duplicated.`);
    }
    ids.add(id);

    const title = typeof raw.title === "string" ? raw.title.trim() : "";
    if (!title) {
      throw new Error(`Entry '${id}' is missing 'title'.`);
    }

    const category = typeof raw.category === "string" ? raw.category.trim() : "";
    if (!assertAllowed(category, KNOWLEDGE_BASE_CATEGORIES)) {
      throw new Error(`Entry '${id}' has invalid 'category'.`);
    }

    const content = asStringArray(raw.content);
    if (content.length < 1 || content.length > 3) {
      throw new Error(`Entry '${id}' must include 1-3 'content' lines.`);
    }

    const actionHint = typeof raw.action_hint === "string" ? raw.action_hint.trim() : "";
    if (!actionHint) {
      throw new Error(`Entry '${id}' is missing 'action_hint'.`);
    }

    const impactArea = typeof raw.impact_area === "string" ? raw.impact_area.trim() : "";
    if (!impactArea) {
      throw new Error(`Entry '${id}' is missing 'impact_area'.`);
    }

    const keywords = asStringArray(raw.keywords);
    if (keywords.length === 0) {
      throw new Error(`Entry '${id}' must include at least one keyword.`);
    }

    const objectiveTags = asStringArray(raw.objective_tags);
    if (
      objectiveTags.length > 0 &&
      objectiveTags.some((tag) => !assertAllowed(tag, KNOWLEDGE_BASE_OBJECTIVES))
    ) {
      throw new Error(`Entry '${id}' has invalid 'objective_tags'.`);
    }

    const updatedAt = typeof raw.updated_at === "string" ? raw.updated_at.trim() : "";
    if (!updatedAt || !assertValidIsoDate(updatedAt)) {
      throw new Error(`Entry '${id}' has invalid 'updated_at'.`);
    }

    const confidence = typeof raw.confidence === "string" ? raw.confidence.trim() : "";
    if (!assertAllowed(confidence, KNOWLEDGE_BASE_CONFIDENCE)) {
      throw new Error(`Entry '${id}' has invalid 'confidence'.`);
    }

    if (typeof raw.active !== "boolean") {
      throw new Error(`Entry '${id}' must include boolean 'active'.`);
    }
    if (raw.active) {
      activeCount += 1;
    }

    entries.push({
      id,
      title,
      category,
      content,
      action_hint: actionHint,
      impact_area: impactArea,
      keywords,
      objective_tags: objectiveTags.length > 0 ? objectiveTags as KnowledgeBaseObjective[] : undefined,
      updated_at: updatedAt,
      confidence,
      active: raw.active
    });
  }

  if (activeCount === 0) {
    throw new Error("Knowledge base contains no active entries.");
  }

  return { version, entries };
}

function prepareEntry(entry: KnowledgeBaseEntry): PreparedKnowledgeEntry {
  const searchBlob = [
    entry.title,
    entry.action_hint,
    entry.impact_area,
    ...entry.content,
    ...entry.keywords
  ].join(" ");
  return {
    entry,
    searchTokens: new Set(tokenize(searchBlob)),
    keywordPhrases: entry.keywords.map((value) => normalizeText(value)),
    updatedAtEpoch: Date.parse(entry.updated_at) || 0
  };
}

export function buildKnowledgeBaseStore(dataset: KnowledgeBaseDataset): KnowledgeBaseStore {
  const activeEntries = dataset.entries.filter((entry) => entry.active);
  return {
    version: dataset.version,
    entries: activeEntries.map(prepareEntry)
  };
}

export async function loadKnowledgeBaseStore(path: string): Promise<KnowledgeBaseStore> {
  const raw = await fs.readFile(path, "utf-8");
  const parsed = JSON.parse(raw) as unknown;
  const dataset = validateKnowledgeBaseDataset(parsed);
  return buildKnowledgeBaseStore(dataset);
}

function inferCategoryPrior(question: string): Partial<Record<KnowledgeBaseCategory, number>> {
  const normalized = normalizeText(question);
  const prior: Partial<Record<KnowledgeBaseCategory, number>> = {};

  const boost = (category: KnowledgeBaseCategory, weight: number): void => {
    prior[category] = (prior[category] ?? 0) + weight;
  };

  if (normalized.includes("algorithm") || normalized.includes("fyp") || normalized.includes("for you")) {
    boost("algorithm", 2.0);
  }
  if (normalized.includes("hook") || normalized.includes("first 2 seconds")) {
    boost("hooks", 1.8);
  }
  if (normalized.includes("retention") || normalized.includes("watch time")) {
    boost("retention", 1.8);
  }
  if (normalized.includes("cta") || normalized.includes("call to action") || normalized.includes("comment")) {
    boost("cta", 1.5);
  }
  if (normalized.includes("creator") || normalized.includes("creators")) {
    boost("creators", 2.4);
  }
  if (normalized.includes("hashtag") || normalized.includes("hashtags") || normalized.includes("#")) {
    boost("hashtags", 2.4);
  }
  if (normalized.includes("format") || normalized.includes("style") || normalized.includes("content type")) {
    boost("content_formats", 1.4);
  }
  if (normalized.includes("perform") || normalized.includes("what works") || normalized.includes("trend")) {
    boost("algorithm", 1.2);
    boost("hooks", 0.8);
    boost("retention", 0.8);
  }

  return prior;
}

function isKnowledgeExplicit(question: string): boolean {
  const normalized = normalizeText(question);
  return (
    normalized.includes("algorithm") ||
    normalized.includes("what performs") ||
    normalized.includes("what works") ||
    normalized.includes("top creator") ||
    normalized.includes("top creators") ||
    normalized.includes("top hashtag") ||
    normalized.includes("top hashtags") ||
    normalized.includes("trend")
  );
}

function normalizeObjective(value?: string): KnowledgeBaseObjective | null {
  const normalized = normalizeText(value || "");
  if (normalized === "engagement") return "engagement";
  if (normalized === "conversion") return "conversion";
  if (normalized === "reach") return "reach";
  if (normalized === "community") return "community";
  return null;
}

export function searchKnowledgeBase(
  store: KnowledgeBaseStore,
  params: KnowledgeBaseSearchParams
): KnowledgeBaseSearchResult {
  const maxResults = Math.max(1, Math.min(10, params.maxResults ?? 3));
  const objective = normalizeObjective(params.objective);
  const questionTokens = tokenize(params.question);
  const questionTokenSet = new Set(questionTokens);
  const categoryPrior = inferCategoryPrior(params.question);
  const explicit = isKnowledgeExplicit(params.question);

  const scored: ScoredKnowledgeEntry[] = store.entries.map((prepared) => {
    let lexicalMatches = 0;
    for (const token of questionTokenSet) {
      if (prepared.searchTokens.has(token)) {
        lexicalMatches += 1;
      }
    }

    let phraseBonus = 0;
    const normalizedQuestion = normalizeText(params.question);
    for (const phrase of prepared.keywordPhrases) {
      if (phrase && normalizedQuestion.includes(phrase)) {
        phraseBonus += 1.6;
      }
    }

    const prior = categoryPrior[prepared.entry.category] ?? 0;
    const objectiveBoost =
      objective && prepared.entry.objective_tags?.includes(objective) ? 0.7 : 0;
    const priorityOffset = params.priority === "low" ? -0.15 : 0;
    const confidenceBoost = prepared.entry.confidence === "high" ? 0.25 : 0;
    const score = lexicalMatches + phraseBonus + prior + objectiveBoost + confidenceBoost + priorityOffset;

    return {
      entry: prepared.entry,
      score,
      updatedAtEpoch: prepared.updatedAtEpoch,
      lexicalMatches
    };
  });

  const filtered = scored
    .filter((item) => {
      if (params.priority === "high" || explicit) {
        return item.score > 0.5;
      }
      return item.lexicalMatches > 0 || item.score > 1.3;
    })
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      return right.updatedAtEpoch - left.updatedAtEpoch;
    })
    .slice(0, maxResults);

  return {
    source: "knowledge_base",
    entries: filtered.map((item) => item.entry),
    matched_categories: Array.from(
      new Set(filtered.map((item) => item.entry.category))
    ) as KnowledgeBaseCategory[],
    priority: params.priority
  };
}
