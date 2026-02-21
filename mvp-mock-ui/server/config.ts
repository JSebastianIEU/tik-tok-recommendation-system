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
