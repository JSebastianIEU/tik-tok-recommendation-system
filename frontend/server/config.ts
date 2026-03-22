import path from "node:path";
import dotenv from "dotenv";

dotenv.config({ path: path.resolve(process.cwd(), ".env.local") });

export const SERVER_PORT = Number(process.env.PORT ?? "5174");
export const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
export const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-reasoner";
export const DEEPSEEK_BASE_URL =
  process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";
export const DEEPSEEK_ENABLED =
  Boolean(DEEPSEEK_API_KEY.trim()) && DEEPSEEK_API_KEY.trim() !== "your_key_here";

export const RECOMMENDER_BASE_URL =
  process.env.RECOMMENDER_BASE_URL?.trim() || "http://127.0.0.1:8081";
export const RECOMMENDER_ENABLED =
  process.env.RECOMMENDER_ENABLED?.trim().toLowerCase() !== "false";
export const RECOMMENDER_TIMEOUT_MS = Number(
  process.env.RECOMMENDER_TIMEOUT_MS ?? "3500"
);
export const RECOMMENDER_RETRY_COUNT = Number(
  process.env.RECOMMENDER_RETRY_COUNT ?? "1"
);
