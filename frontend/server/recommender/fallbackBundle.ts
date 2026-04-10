import fs from "node:fs/promises";
import path from "node:path";

export interface FallbackBundleItem {
  candidate_id: string;
  score: number;
}

export interface FallbackBundle {
  objective: string;
  generated_at: string;
  version?: string;
  items: FallbackBundleItem[];
}

interface CacheEntry {
  loadedAt: number;
  bundle: FallbackBundle | null;
}

export class FallbackBundleStore {
  private readonly rootDir: string;
  private readonly ttlMs: number;
  private readonly cache = new Map<string, CacheEntry>();

  constructor(rootDir: string, ttlMs: number) {
    this.rootDir = rootDir;
    this.ttlMs = Math.max(1000, ttlMs);
  }

  private key(objective: string): string {
    const normalized = objective.trim().toLowerCase() || "engagement";
    return normalized === "community" ? "engagement" : normalized;
  }

  private async readBundle(objective: string): Promise<FallbackBundle | null> {
    const normalized = this.key(objective);
    const filePath = path.resolve(process.cwd(), this.rootDir, `${normalized}.json`);
    try {
      const raw = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(raw) as Partial<FallbackBundle>;
      if (!parsed || !Array.isArray(parsed.items)) {
        return null;
      }
      const items = parsed.items
        .map((item) => {
          if (!item || typeof item !== "object") {
            return null;
          }
          const candidate = (item as { candidate_id?: unknown; score?: unknown }).candidate_id;
          const score = (item as { candidate_id?: unknown; score?: unknown }).score;
          if (typeof candidate !== "string" || typeof score !== "number" || !Number.isFinite(score)) {
            return null;
          }
          return { candidate_id: candidate, score };
        })
        .filter((item): item is FallbackBundleItem => Boolean(item));
      return {
        objective: normalized,
        generated_at:
          typeof parsed.generated_at === "string" ? parsed.generated_at : new Date().toISOString(),
        version: typeof parsed.version === "string" ? parsed.version : "fallback_bundle.v1",
        items
      };
    } catch {
      return null;
    }
  }

  async get(objective: string, now = Date.now()): Promise<FallbackBundle | null> {
    const key = this.key(objective);
    const cached = this.cache.get(key);
    if (cached && now - cached.loadedAt <= this.ttlMs) {
      return cached.bundle;
    }
    const loaded = await this.readBundle(key);
    this.cache.set(key, { loadedAt: now, bundle: loaded });
    return loaded;
  }
}
