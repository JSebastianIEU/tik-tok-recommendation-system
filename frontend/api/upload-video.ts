import type { VercelRequest, VercelResponse } from "@vercel/node";
import { randomUUID } from "crypto";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL =
  process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";

interface UploadJsonBody {
  file_name?: string;
  file_type?: string;
  file_size?: number;
}

function collectBody(req: VercelRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    req.on("data", (chunk: Uint8Array) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

async function generateCaption(fileName: string): Promise<string> {
  if (!DEEPSEEK_API_KEY) return "";

  try {
    const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        temperature: 0.5,
        max_tokens: 200,
        messages: [
          {
            role: "system",
            content:
              "You are a TikTok content strategist. Given a video file name, " +
              "write a short, engaging TikTok caption (1-2 sentences) that guesses " +
              "what the video might be about based on the file name. Be creative but plausible. " +
              "Do NOT include hashtags. No emojis. Plain text only.",
          },
          { role: "user", content: `File name: ${fileName}` },
        ],
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!resp.ok) return "";
    const data = (await resp.json()) as {
      choices: { message: { content: string } }[];
    };
    return data.choices?.[0]?.message?.content?.trim() ?? "";
  } catch {
    return "";
  }
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, x-file-name, x-file-type"
  );

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  // Support both JSON body (new) and raw octet-stream with headers (legacy)
  let fileName = (req.headers["x-file-name"] as string) ?? "uploaded_video.mp4";
  let fileType = (req.headers["x-file-type"] as string) ?? "video/mp4";

  const contentType = (req.headers["content-type"] ?? "").toString().toLowerCase();

  if (contentType.includes("application/json")) {
    // New path: small JSON payload with metadata only
    try {
      const raw = await collectBody(req);
      const jsonBody = JSON.parse(raw.toString("utf-8")) as UploadJsonBody;
      if (jsonBody.file_name) fileName = jsonBody.file_name;
      if (jsonBody.file_type) fileType = jsonBody.file_type;
    } catch {
      return res.status(400).json({ error: "Invalid JSON body." });
    }
  } else {
    // Legacy path: consume raw binary body (not stored on Vercel)
    try {
      await collectBody(req);
    } catch {
      return res.status(400).json({ error: "Failed to read upload." });
    }
  }

  const assetId = randomUUID();
  const caption = await generateCaption(fileName);

  // Return a minimal analysis result matching VideoAnalysisResult
  return res.status(201).json({
    asset_id: assetId,
    file_name: fileName,
    file_type: fileType,
    duration_seconds: null,
    video_caption: caption || null,
    transcript: null,
    ocr_text: null,
    visual_features: null,
    timeline: null,
    asset: {
      asset_id: assetId,
      original_filename: fileName,
      content_type: fileType,
      size_bytes: 0,
      duration_seconds: null,
    },
  });
}
