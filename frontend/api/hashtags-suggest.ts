import type { VercelRequest, VercelResponse } from "@vercel/node";

const RECOMMENDER_SERVICE_URL = process.env.RECOMMENDER_SERVICE_URL ?? "";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  if (!RECOMMENDER_SERVICE_URL) {
    return res.status(503).json({ error: "Recommender service not configured" });
  }

  try {
    const body = req.body as Record<string, unknown> | null;
    const caption = typeof body?.caption === "string" ? body.caption : "";
    const topN = typeof body?.top_n === "number" ? body.top_n : 15;
    const excludeTags = Array.isArray(body?.exclude_tags) ? body.exclude_tags : [];
    const includeNeighbours = body?.include_neighbours === true;

    const resp = await fetch(`${RECOMMENDER_SERVICE_URL}/v1/hashtags/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        caption,
        top_n: topN,
        exclude_tags: excludeTags,
        include_neighbours: includeNeighbours,
      }),
      signal: AbortSignal.timeout(25000),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      console.error("[hashtags-suggest] upstream error:", resp.status, text);
      return res.status(502).json({ error: "Upstream recommender error" });
    }

    const data = (await resp.json()) as Record<string, unknown>;

    // Map the Python response format to the frontend expected format
    // Python returns: { hashtags: [{hashtag, score, frequency, avg_engagement}], neighbours: [...] }
    // Frontend expects: { suggestions: [{hashtag, score, frequency, avg_engagement}], neighbours: [...] }
    const hashtags = Array.isArray(data.hashtags) ? data.hashtags : [];
    const neighbours = Array.isArray(data.neighbours) ? data.neighbours : [];

    return res.status(200).json({
      suggestions: hashtags,
      neighbours,
    });
  } catch (err) {
    console.error("[hashtags-suggest] error:", err);
    return res.status(500).json({ error: "Internal error" });
  }
}
