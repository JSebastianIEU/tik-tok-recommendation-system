import type { VercelRequest, VercelResponse } from "@vercel/node";

export const config = { maxDuration: 10 };

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method === "OPTIONS") return res.status(204).end();

  const url = (req.query.url as string) || "";
  if (!url) return res.status(400).json({ error: "Missing url parameter" });

  try {
    const resp = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        Referer: "https://www.tiktok.com/",
      },
      signal: AbortSignal.timeout(8000),
    });

    if (!resp.ok) return res.status(resp.status).end();

    const contentType = resp.headers.get("content-type") || "image/jpeg";
    const buffer = Buffer.from(await resp.arrayBuffer());

    res.setHeader("Content-Type", contentType);
    res.setHeader("Cache-Control", "public, max-age=86400, s-maxage=86400");
    return res.send(buffer);
  } catch {
    return res.status(502).json({ error: "Failed to fetch thumbnail" });
  }
}
