import type { VercelRequest, VercelResponse } from "@vercel/node";
import { readFileSync } from "fs";
import { join } from "path";
import { Client as PgClient } from "pg";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL =
  process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";
const RECOMMENDER_SERVICE_URL = process.env.RECOMMENDER_SERVICE_URL ?? "";
const DATABASE_URL = process.env.DATABASE_URL ?? "";

export const config = { maxDuration: 60 };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface DemoRecord {
  video_id: string;
  video_url: string;
  caption: string;
  hashtags: string[];
  keywords?: string[];
  likes: number;
  comments_count: number;
  shares: number;
  views: number;
  author?: { nickname?: string; unique_id?: string };
}

let cachedRecords: DemoRecord[] | null = null;

function loadDemoData(): DemoRecord[] {
  if (cachedRecords) return cachedRecords;
  try {
    const raw = readFileSync(
      join(process.cwd(), "frontend", "src", "data", "demodata.jsonl"),
      "utf-8"
    );
    cachedRecords = raw
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as DemoRecord);
    return cachedRecords;
  } catch {
    return [];
  }
}

// Real ranker weights from trained model
const RANKER_WEIGHTS: Record<
  string,
  Record<string, number>
> = {
  reach: {
    semantic_relevance: 0.28,
    intent_alignment: 0.22,
    performance_quality: 0.25,
    reference_usefulness: 0.15,
    support_confidence: 0.1,
  },
  engagement: {
    semantic_relevance: 0.28,
    intent_alignment: 0.25,
    performance_quality: 0.22,
    reference_usefulness: 0.15,
    support_confidence: 0.1,
  },
  conversion: {
    semantic_relevance: 0.26,
    intent_alignment: 0.3,
    performance_quality: 0.2,
    reference_usefulness: 0.14,
    support_confidence: 0.1,
  },
};

function objectiveWeights(obj: string) {
  return RANKER_WEIGHTS[obj] ?? RANKER_WEIGHTS.engagement;
}

function textOverlap(a: string[], b: string[]): number {
  const setA = new Set(a.map((t) => t.toLowerCase().replace(/^[#@]/, "")));
  const setB = new Set(b.map((t) => t.toLowerCase().replace(/^[#@]/, "")));
  if (setA.size === 0 && setB.size === 0) return 0.5;
  let overlap = 0;
  for (const t of setA) if (setB.has(t)) overlap++;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0.5 : overlap / union;
}

function cosineSimilarityWords(a: string, b: string): number {
  const wordsA = a.toLowerCase().split(/\W+/).filter(Boolean);
  const wordsB = b.toLowerCase().split(/\W+/).filter(Boolean);
  const vocab = new Set([...wordsA, ...wordsB]);
  if (vocab.size === 0) return 0;
  let dot = 0,
    magA = 0,
    magB = 0;
  for (const w of vocab) {
    const cA = wordsA.filter((x) => x === w).length;
    const cB = wordsB.filter((x) => x === w).length;
    dot += cA * cB;
    magA += cA * cA;
    magB += cB * cB;
  }
  return magA === 0 || magB === 0 ? 0 : dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function scoreCandidate(
  rec: DemoRecord,
  description: string,
  hashtags: string[],
  objective: string
) {
  const w = objectiveWeights(objective);
  const semantic = cosineSimilarityWords(description, rec.caption);
  const hashtagSim = textOverlap(hashtags, rec.hashtags ?? []);
  const intent = hashtagSim * 0.6 + semantic * 0.4;
  const maxViews = 1e7;
  const perf = Math.min(
    1,
    (Math.log10((rec.views || 1) + 1) / Math.log10(maxViews + 1)) * 0.5 +
      (rec.likes / Math.max(rec.views, 1)) * 5
  );
  const ref = semantic > 0.3 ? 0.7 + semantic * 0.3 : 0.4;
  const support = hashtagSim > 0.2 ? 0.7 : 0.4;

  const score =
    w.semantic_relevance * semantic +
    w.intent_alignment * intent +
    w.performance_quality * perf +
    w.reference_usefulness * ref +
    w.support_confidence * support;

  return {
    score: Math.round(score * 1000) / 1000,
    components: {
      semantic_relevance: Math.round(semantic * 1000) / 1000,
      intent_alignment: Math.round(intent * 1000) / 1000,
      performance_quality: Math.round(perf * 1000) / 1000,
      reference_usefulness: Math.round(ref * 1000) / 1000,
      support_confidence: Math.round(support * 1000) / 1000,
    },
  };
}

function engRate(rec: DemoRecord): string {
  const v = rec.views || 1;
  return (((rec.likes + rec.comments_count + rec.shares) / v) * 100).toFixed(2) + "%";
}

function buildComparable(
  rec: DemoRecord,
  idx: number,
  description: string,
  hashtags: string[],
  objective: string
) {
  const scored = scoreCandidate(rec, description, hashtags, objective);
  const authorName =
    rec.author?.nickname ?? rec.author?.unique_id ?? "unknown";
  const matchedKw = hashtags.filter((h) =>
    (rec.hashtags ?? []).some(
      (rh) => rh.toLowerCase().replace("#", "") === h.toLowerCase().replace("#", "")
    )
  );

  return {
    id: `comp-${idx}`,
    candidate_id: rec.video_id,
    caption: rec.caption,
    author: authorName,
    video_url: rec.video_url ?? "",
    thumbnail_url: "",
    hashtags: rec.hashtags ?? [],
    similarity: scored.score,
    support_level: scored.score > 0.5 ? "full" : scored.score > 0.3 ? "partial" : "low",
    confidence_label:
      scored.score > 0.5
        ? "High confidence"
        : scored.score > 0.3
        ? "Medium confidence"
        : "Low confidence",
    metrics: {
      views: rec.views ?? 0,
      likes: rec.likes ?? 0,
      comments_count: rec.comments_count ?? 0,
      shares: rec.shares ?? 0,
      engagement_rate: engRate(rec),
    },
    matched_keywords: matchedKw,
    observations: [],
    why_this_was_chosen: `Ranked #${idx + 1} by ${objective} objective scoring.`,
    ranking_reasons: [
      `Semantic relevance: ${scored.components.semantic_relevance}`,
      `Intent alignment: ${scored.components.intent_alignment}`,
      `Performance quality: ${scored.components.performance_quality}`,
    ],
    score_components: scored.components,
    retrieval_branches: ["bm25_text"],
  };
}

// ---------------------------------------------------------------------------
// Build a full ReportOutput
// ---------------------------------------------------------------------------

function buildReport(
  payload: {
    description: string;
    hashtags: string[];
    mentions: string[];
    objective: string;
    content_type?: string;
  },
  comparables: ReturnType<typeof buildComparable>[]
) {
  const now = new Date().toISOString();
  const reqId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  const avgScore =
    comparables.reduce((s, c) => s + c.similarity, 0) / Math.max(comparables.length, 1);
  const avgEng =
    comparables.reduce(
      (s, c) => s + parseFloat(c.metrics.engagement_rate),
      0
    ) / Math.max(comparables.length, 1);
  const topHashtags = Array.from(
    new Set(comparables.flatMap((c) => c.hashtags))
  ).slice(0, 8);

  const scoreComponentAvgs = {
    semantic_relevance: 0,
    intent_alignment: 0,
    performance_quality: 0,
    reference_usefulness: 0,
    support_confidence: 0,
  };
  for (const c of comparables) {
    for (const k of Object.keys(scoreComponentAvgs) as (keyof typeof scoreComponentAvgs)[]) {
      scoreComponentAvgs[k] += c.score_components[k] / comparables.length;
    }
  }
  for (const k of Object.keys(scoreComponentAvgs) as (keyof typeof scoreComponentAvgs)[]) {
    scoreComponentAvgs[k] = Math.round(scoreComponentAvgs[k] * 1000) / 1000;
  }

  return {
    meta: {
      request_id: reqId,
      objective: payload.objective,
      objective_effective: payload.objective === "community" ? "engagement" : payload.objective,
      generated_at: now,
      recommender_source: "deterministic-local" as const,
      fallback_mode: false,
      fallback_reason: null,
      evidence_label: comparables.length >= 8 ? "Strong evidence" as const : "Moderate evidence" as const,
      confidence_label: avgScore > 0.4 ? "High confidence" as const : "Medium confidence" as const,
      experiment_id: null,
      variant: null,
    },
    header: {
      title: "TikTok Content Analysis Report",
      subtitle: `${payload.objective} optimization | ${comparables.length} comparable videos`,
      badges: {
        candidates_k: comparables.length,
        model: "ranker-v2",
        mode: payload.objective,
      },
      disclaimer:
        "Scores are computed using trained ranker weights. Results may vary.",
    },
    executive_summary: {
      metrics: [
        { id: "avg-score", label: "Avg Relevance Score", value: avgScore.toFixed(3) },
        { id: "avg-eng", label: "Avg Engagement Rate", value: avgEng.toFixed(2) + "%" },
        { id: "candidates", label: "Videos Analyzed", value: String(comparables.length) },
        { id: "objective", label: "Optimization Target", value: payload.objective },
      ],
      extracted_keywords: payload.hashtags.map((h) => h.replace("#", "")),
      meaning_points: [
        `Top comparable videos show an average engagement rate of ${avgEng.toFixed(2)}%.`,
        `Content in this niche is optimized for ${payload.objective}.`,
        `${topHashtags.length} recurring hashtags identified across comparables.`,
      ],
      summary_text: `Analysis of ${comparables.length} comparable TikTok videos for ${payload.objective} optimization. The top-ranked video scored ${comparables[0]?.similarity.toFixed(3) ?? "N/A"} in relevance.`,
    },
    comparables,
    direct_comparison: {
      rows: [
        {
          id: "dc-sem",
          label: "Semantic Relevance",
          your_value_label: scoreComponentAvgs.semantic_relevance.toFixed(2),
          comparable_value_label: "baseline",
          your_value_pct: scoreComponentAvgs.semantic_relevance * 100,
          comparable_value_pct: 50,
        },
        {
          id: "dc-intent",
          label: "Intent Alignment",
          your_value_label: scoreComponentAvgs.intent_alignment.toFixed(2),
          comparable_value_label: "baseline",
          your_value_pct: scoreComponentAvgs.intent_alignment * 100,
          comparable_value_pct: 50,
        },
      ],
      note: "Comparison against corpus average baseline.",
    },
    relevant_comments: { items: [], disclaimer: "Comment analysis not available in this mode." },
    recommendations: {
      items: [
        {
          id: "rec-1",
          title: `Optimize for ${payload.objective}`,
          priority: "High" as const,
          effort: "Medium" as const,
          evidence: `Based on ${comparables.length} comparable videos`,
          rationale: `Top-performing videos in this niche use hashtags like ${topHashtags.slice(0, 3).join(", ")}`,
          confidence_label: "High confidence" as const,
          effect_area: "topic_alignment" as const,
          caveats: [],
          evidence_refs: comparables.slice(0, 3).map((c) => c.candidate_id),
        },
      ],
    },
    reasoning: {
      evidence_pack: {
        version: "1.0",
        request: {
          request_id: reqId,
          objective: payload.objective,
          objective_effective: payload.objective === "community" ? "engagement" : payload.objective,
          fallback_mode: false,
        },
        query_summary: {
          description: payload.description,
          hashtags: payload.hashtags,
          mentions: payload.mentions,
          content_type: payload.content_type,
        },
        candidate_summary: {
          final_count: comparables.length,
          top_k_considered: comparables.length,
          support_mix: {
            full: comparables.filter((c) => c.support_level === "full").length,
            partial: comparables.filter((c) => c.support_level === "partial").length,
            low: comparables.filter((c) => c.support_level === "low").length,
          },
          branch_mix: { bm25_text: comparables.length },
        },
        top_candidates: comparables.slice(0, 5).map((c, i) => ({
          candidate_id: c.candidate_id,
          rank: i + 1,
          score: c.similarity,
          support_level: c.support_level,
          score_components: c.score_components,
          ranking_reasons: c.ranking_reasons,
          hashtags: c.hashtags,
        })),
        aggregate_patterns: {
          repeated_hashtags: topHashtags.map((t) => ({
            tag: t,
            support_count: comparables.filter((c) => c.hashtags.includes(t)).length,
          })),
          repeated_content_types: [],
          repeated_ranking_reasons: [],
          score_component_averages: scoreComponentAvgs,
        },
        contrast_signals: { top_vs_rest: [], mismatches: [], conflicts: [] },
        evidence_quality: {
          sufficient: comparables.length >= 5,
          confidence: avgScore,
          missing_flags: [],
        },
      },
      explanation_units: [
        {
          explanation_id: "exp-1",
          claim_type: "pattern_summary" as const,
          statement: `Videos in this niche average ${avgEng.toFixed(2)}% engagement rate.`,
          evidence_refs: comparables.slice(0, 3).map((c) => c.candidate_id),
          confidence: avgScore,
          status: avgScore > 0.4 ? "strong" as const : "moderate" as const,
          caveats: [],
        },
      ],
      recommendation_units: [
        {
          recommendation_id: "runit-1",
          action: `Use trending hashtags: ${topHashtags.slice(0, 5).join(", ")}`,
          rationale: "These hashtags appear most frequently among high-scoring comparables.",
          priority: "High" as const,
          effort: "Low" as const,
          confidence: 0.8,
          evidence_refs: comparables.slice(0, 3).map((c) => c.candidate_id),
          expected_effect_area: "topic_alignment" as const,
          caveats: [],
        },
      ],
      reasoning_metadata: {
        version: "1.0",
        fallback_mode: false,
        evidence_sufficiency: comparables.length >= 5,
        reasoning_confidence: avgScore,
        missing_evidence_flags: [],
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Supabase enrichment — fetch real engagement metrics for candidate IDs
// ---------------------------------------------------------------------------

interface VideoEnrichment {
  caption: string;
  author_username: string;
  author_id: string;
  thumbnail_url: string;
  video_url: string;
  views: number;
  likes: number;
  comments_count: number;
  shares: number;
}

async function enrichFromSupabase(
  candidateIds: string[]
): Promise<Map<string, VideoEnrichment>> {
  const map = new Map<string, VideoEnrichment>();
  if (!DATABASE_URL || candidateIds.length === 0) return map;

  const client = new PgClient({
    connectionString: DATABASE_URL,
    ssl: { rejectUnauthorized: false },
  });
  try {
    await client.connect();

    const sql = `
      SELECT
        v.video_id,
        v.caption,
        v.author_id,
        v.thumbnail_url,
        v.url AS video_url,
        a.username AS author_username,
        s.plays   AS views,
        s.likes,
        s.comments_count,
        s.shares
      FROM videos v
      LEFT JOIN authors a ON a.author_id = v.author_id
      LEFT JOIN LATERAL (
        SELECT vs.plays, vs.likes, vs.comments_count, vs.shares
        FROM video_snapshots vs
        WHERE vs.video_id = v.video_id
        ORDER BY vs.scraped_at DESC
        LIMIT 1
      ) s ON true
      WHERE v.video_id = ANY($1)
    `;

    const result = await client.query(sql, [candidateIds]);

    for (const row of result.rows) {
      map.set(row.video_id, {
        caption: row.caption ?? "",
        author_username: row.author_username ?? row.author_id ?? "unknown",
        author_id: row.author_id ?? "",
        thumbnail_url: row.thumbnail_url ?? "",
        video_url: row.video_url ?? "",
        views: Number(row.views) || 0,
        likes: Number(row.likes) || 0,
        comments_count: Number(row.comments_count) || 0,
        shares: Number(row.shares) || 0,
      });
    }
  } catch (err) {
    console.error("[enrichFromSupabase] error:", err);
  } finally {
    await client.end().catch(() => {});
  }

  return map;
}

// ---------------------------------------------------------------------------
// Supabase candidate search — find relevant videos by hashtag + text match
// ---------------------------------------------------------------------------

async function searchCandidatesFromSupabase(
  description: string,
  hashtags: string[],
  limit: number = 50
): Promise<DemoRecord[]> {
  if (!DATABASE_URL) return [];

  const client = new PgClient({
    connectionString: DATABASE_URL,
    ssl: { rejectUnauthorized: false },
  });
  try {
    await client.connect();

    // Build search terms from hashtags and description keywords
    const hashtagTerms = hashtags
      .map((h) => h.toLowerCase().replace(/^#/, "").trim())
      .filter(Boolean);
    const descWords = description
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length >= 3);
    const searchTerms = [...new Set([...hashtagTerms, ...descWords.slice(0, 10)])];

    if (searchTerms.length === 0) {
      // No search terms — return recent high-engagement videos
      const fallbackSql = `
        SELECT
          v.video_id, v.caption, v.author_id, v.thumbnail_url,
          v.url AS video_url,
          a.username AS author_username,
          s.plays AS views, s.likes, s.comments_count, s.shares,
          ARRAY(
            SELECT h.tag FROM video_hashtags vh
            JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
            WHERE vh.video_id = v.video_id
          ) AS hashtags
        FROM videos v
        LEFT JOIN authors a ON a.author_id = v.author_id
        LEFT JOIN LATERAL (
          SELECT vs.plays, vs.likes, vs.comments_count, vs.shares
          FROM video_snapshots vs WHERE vs.video_id = v.video_id
          ORDER BY vs.scraped_at DESC LIMIT 1
        ) s ON true
        WHERE s.plays > 1000
        ORDER BY s.plays DESC
        LIMIT $1
      `;
      const result = await client.query(fallbackSql, [limit]);
      return result.rows.map(rowToRecord);
    }

    // Search by hashtag match first, then caption text match
    const sql = `
      WITH hashtag_matches AS (
        SELECT DISTINCT vh.video_id, 2 AS match_weight
        FROM video_hashtags vh
        JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
        WHERE LOWER(h.tag) = ANY($1)
      ),
      caption_matches AS (
        SELECT v.video_id, 1 AS match_weight
        FROM videos v
        WHERE ${searchTerms.map((_, i) => `LOWER(v.caption) LIKE $${i + 3}`).join(" OR ")}
      ),
      all_matches AS (
        SELECT video_id, SUM(match_weight) AS relevance
        FROM (
          SELECT * FROM hashtag_matches
          UNION ALL
          SELECT * FROM caption_matches
        ) combined
        GROUP BY video_id
      )
      SELECT
        v.video_id, v.caption, v.author_id, v.thumbnail_url,
        v.url AS video_url,
        a.username AS author_username,
        s.plays AS views, s.likes, s.comments_count, s.shares,
        am.relevance,
        ARRAY(
          SELECT h.tag FROM video_hashtags vh
          JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
          WHERE vh.video_id = v.video_id
        ) AS hashtags
      FROM all_matches am
      JOIN videos v ON v.video_id = am.video_id
      LEFT JOIN authors a ON a.author_id = v.author_id
      LEFT JOIN LATERAL (
        SELECT vs.plays, vs.likes, vs.comments_count, vs.shares
        FROM video_snapshots vs WHERE vs.video_id = v.video_id
        ORDER BY vs.scraped_at DESC LIMIT 1
      ) s ON true
      ORDER BY am.relevance DESC, COALESCE(s.plays, 0) DESC
      LIMIT $2
    `;

    const likeTerms = searchTerms.map((t) => `%${t}%`);
    const params: (string | string[] | number)[] = [hashtagTerms, limit, ...likeTerms];
    const result = await client.query(sql, params);
    return result.rows.map(rowToRecord);
  } catch (err) {
    console.error("[searchCandidatesFromSupabase] error:", err);
    return [];
  } finally {
    await client.end().catch(() => {});
  }
}

function rowToRecord(row: Record<string, unknown>): DemoRecord {
  const views = Number(row.views) || 0;
  const likes = Number(row.likes) || 0;
  const comments_count = Number(row.comments_count) || 0;
  const shares = Number(row.shares) || 0;
  return {
    video_id: String(row.video_id ?? ""),
    video_url: String(row.video_url ?? ""),
    caption: String(row.caption ?? ""),
    hashtags: Array.isArray(row.hashtags) ? row.hashtags.map(String) : [],
    keywords: [],
    likes,
    comments_count,
    shares,
    views,
    author: {
      nickname: String(row.author_username ?? row.author_id ?? "unknown"),
      unique_id: String(row.author_username ?? row.author_id ?? "unknown"),
    },
  };
}

// ---------------------------------------------------------------------------
// Try to call Cloud Run recommender service
// ---------------------------------------------------------------------------

async function callRecommenderService(payload: Record<string, unknown>) {
  if (!RECOMMENDER_SERVICE_URL) {
    console.warn("[callRecommenderService] RECOMMENDER_SERVICE_URL is empty — set this env var to your Cloud Run URL");
    return null;
  }

  const description = typeof payload.description === "string" ? payload.description : "";
  const hashtags = Array.isArray(payload.hashtags) ? payload.hashtags as string[] : [];
  const mentions = Array.isArray(payload.mentions) ? payload.mentions as string[] : [];
  const objective = typeof payload.objective === "string" ? payload.objective : "engagement";
  const contentType = typeof payload.content_type === "string" ? payload.content_type : "video";

  try {
    console.log("[callRecommenderService] calling", RECOMMENDER_SERVICE_URL);
    const recResp = await fetch(`${RECOMMENDER_SERVICE_URL}/v1/recommendations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: { text: description, hashtags, mentions, content_type: contentType },
        objective,
        top_k: 10,
        as_of_time: new Date().toISOString(),
        routing: { stage_budgets_ms: { retrieval: 60000, ranking: 60000, explainability: 60000 } },
      }),
      signal: AbortSignal.timeout(55000),
    });
    if (!recResp.ok) {
      console.error("[callRecommenderService] HTTP", recResp.status, await recResp.text().catch(() => ""));
      return null;
    }
    const recData = await recResp.json() as {
      items?: Array<{
        candidate_id: string; score: number; caption?: string; hashtags?: string[];
        author_id?: string; score_components?: Record<string, number>;
        ranking_reasons?: string[]; retrieval_branch_scores?: Record<string, number>;
        views?: number; likes?: number; comments_count?: number; shares?: number;
        engagement_rate?: number;
      }>;
    };
    const items = recData.items ?? [];
    if (items.length === 0) return null;

    // Enrich with real Supabase data (views, likes, captions, author names)
    const candidateIds = items.map((it) => it.candidate_id);
    const enrichment = await enrichFromSupabase(candidateIds);

    const comparables = items.map((item, idx) => {
      const enriched = enrichment.get(item.candidate_id);
      const v = enriched?.views ?? item.views ?? 0;
      const l = enriched?.likes ?? item.likes ?? 0;
      const cc = enriched?.comments_count ?? item.comments_count ?? 0;
      const s = enriched?.shares ?? item.shares ?? 0;
      const er = v > 0 ? (l + cc + s) / v : 0;
      return {
        id: `comp-${idx}`,
        candidate_id: item.candidate_id,
        caption: enriched?.caption || item.caption || "",
        author: enriched?.author_username || item.author_id || "unknown",
        video_url: enriched?.video_url || "",
        thumbnail_url: enriched?.thumbnail_url || "",
        hashtags: item.hashtags ?? [],
        similarity: item.score,
        support_level: item.score > 0.5 ? "full" : item.score > 0.3 ? "partial" : "low",
        confidence_label: item.score > 0.5 ? ("High confidence" as const) : item.score > 0.3 ? ("Medium confidence" as const) : ("Low confidence" as const),
        metrics: {
          views: v,
          likes: l,
          comments_count: cc,
          shares: s,
          engagement_rate: `${(er * 100).toFixed(2)}%`,
        },
        matched_keywords: hashtags.filter(h => (item.hashtags ?? []).some(rh => rh.toLowerCase().replace("#", "") === h.toLowerCase().replace("#", ""))),
        observations: [] as string[],
        why_this_was_chosen: `Ranked #${idx + 1} by ${objective} objective scoring.`,
        ranking_reasons: item.ranking_reasons ?? [],
        score_components: item.score_components ?? {},
        retrieval_branches: Object.keys(item.retrieval_branch_scores ?? {}),
      };
    });

    const report = buildReport(
      { description, hashtags, mentions, objective, content_type: contentType },
      comparables as Parameters<typeof buildReport>[1]
    );
    (report.meta as Record<string, unknown>).recommender_source = "cloud-run-python";

    return { report, suggested_hashtags: [] };
  } catch (err) {
    console.error("[callRecommenderService] error:", err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Optionally enrich with DeepSeek
// ---------------------------------------------------------------------------

async function enrichSummaryWithLLM(report: ReturnType<typeof buildReport>) {
  if (!DEEPSEEK_API_KEY) return report;

  try {
    const top3 = report.comparables.slice(0, 3);
    const prompt = [
      "Given this TikTok content analysis data, write a 2-3 sentence executive summary and 3 actionable recommendations.",
      `Objective: ${report.meta.objective}`,
      `Top comparable videos:`,
      ...top3.map(
        (c, i) =>
          `${i + 1}. "${c.caption}" (score: ${c.similarity}, engagement: ${c.metrics.engagement_rate}, hashtags: ${c.hashtags.join(", ")})`
      ),
      "",
      "Return JSON: { summary_text: string, recommendations: [{ title: string, rationale: string }] }",
      "No markdown, no emojis, plain JSON only.",
    ].join("\n");

    const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        ...(DEEPSEEK_MODEL.includes("reasoner") ? {} : { temperature: 0.2 }),
        max_tokens: 1024,
        messages: [
          ...(DEEPSEEK_MODEL.includes("reasoner")
            ? []
            : [{ role: "system" as const, content: "You are a senior growth analyst. Return valid JSON only. No markdown." }]),
          { role: "user", content: DEEPSEEK_MODEL.includes("reasoner")
            ? "You are a senior growth analyst. Return valid JSON only. No markdown.\n\n" + prompt
            : prompt },
        ],
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!resp.ok) return report;

    const data = (await resp.json()) as {
      choices: { message: { content: string } }[];
    };
    const raw = data.choices?.[0]?.message?.content?.trim() ?? "";
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return report;

    const parsed = JSON.parse(jsonMatch[0]) as {
      summary_text?: string;
      recommendations?: { title: string; rationale: string }[];
    };

    if (parsed.summary_text) {
      report.executive_summary.summary_text = parsed.summary_text;
    }
    if (parsed.recommendations && Array.isArray(parsed.recommendations)) {
      for (let i = 0; i < Math.min(parsed.recommendations.length, 3); i++) {
        const r = parsed.recommendations[i];
        if (report.recommendations.items[i]) {
          report.recommendations.items[i].title = r.title;
          report.recommendations.items[i].rationale = r.rationale;
        } else {
          report.recommendations.items.push({
            id: `rec-llm-${i}`,
            title: r.title,
            priority: "Medium" as const,
            effort: "Medium" as const,
            evidence: `LLM-generated from ${top3.length} comparables`,
            rationale: r.rationale,
            confidence_label: "Medium confidence" as const,
            effect_area: "topic_alignment" as const,
            caveats: [],
            evidence_refs: top3.map((c) => c.candidate_id),
          });
        }
      }
    }
  } catch {
    // LLM enrichment is best-effort
  }

  return report;
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const body = req.body as Record<string, unknown>;
  const description =
    typeof body.description === "string" ? body.description.trim() : "";
  const hashtags = Array.isArray(body.hashtags)
    ? (body.hashtags as string[])
    : [];
  const mentions = Array.isArray(body.mentions)
    ? (body.mentions as string[])
    : [];
  const objective =
    typeof body.objective === "string" ? body.objective : "engagement";
  const contentType =
    typeof body.content_type === "string" ? body.content_type : undefined;

  // 1) Try Cloud Run recommender service (best quality)
  const serviceResult = await callRecommenderService(body);
  if (serviceResult) {
    return res.json(serviceResult);
  }

  // 2) Try Supabase direct search (real corpus, topically relevant)
  let records = await searchCandidatesFromSupabase(description, hashtags, 50);
  let candidateSource = "supabase";

  // 3) Last resort: static demodata
  if (records.length === 0) {
    records = loadDemoData();
    candidateSource = "demodata";
    console.warn("[generate-report] Supabase returned 0 candidates, falling back to demodata.jsonl");
  }

  if (records.length === 0) {
    return res.status(500).json({ error: "No candidate data available. Set DATABASE_URL or RECOMMENDER_SERVICE_URL." });
  }

  const scored = records.map((rec, idx) =>
    buildComparable(rec, idx, description, hashtags, objective)
  );
  scored.sort((a, b) => b.similarity - a.similarity);

  // Filter out very low relevance candidates — don't return garbage matches
  const MIN_SCORE = 0.15;
  const relevant = scored.filter((c) => c.similarity >= MIN_SCORE);
  const top = (relevant.length >= 3 ? relevant : scored).slice(0, 10);

  let report = buildReport(
    { description, hashtags, mentions, objective, content_type: contentType },
    top
  );

  // Tag the report source so the UI knows where candidates came from
  (report.meta as Record<string, unknown>).recommender_source = candidateSource;
  if (candidateSource !== "supabase") {
    (report.meta as Record<string, unknown>).fallback_mode = true;
    (report.meta as Record<string, unknown>).fallback_reason =
      !RECOMMENDER_SERVICE_URL ? "RECOMMENDER_SERVICE_URL not set" :
      !DATABASE_URL ? "DATABASE_URL not set" : "all_sources_failed";
  }

  report = await enrichSummaryWithLLM(report);

  return res.json({ report, suggested_hashtags: [] });
}
