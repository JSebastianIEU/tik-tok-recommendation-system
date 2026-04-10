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
export const RECOMMENDER_COMPAT_CHECK_INTERVAL_MS = Number(
  process.env.RECOMMENDER_COMPAT_CHECK_INTERVAL_MS ?? "30000"
);
export const RECOMMENDER_COMPAT_CACHE_TTL_MS = Number(
  process.env.RECOMMENDER_COMPAT_CACHE_TTL_MS ?? "20000"
);
export const RECOMMENDER_FALLBACK_BUNDLE_DIR =
  process.env.RECOMMENDER_FALLBACK_BUNDLE_DIR?.trim() || "artifacts/recommender_fallback";
export const RECOMMENDER_FALLBACK_CACHE_TTL_MS = Number(
  process.env.RECOMMENDER_FALLBACK_CACHE_TTL_MS ?? "60000"
);
export const RECOMMENDER_BUDGET_PARSE_MS = Number(
  process.env.RECOMMENDER_BUDGET_PARSE_MS ?? "60"
);
export const RECOMMENDER_BUDGET_NETWORK_MS = Number(
  process.env.RECOMMENDER_BUDGET_NETWORK_MS ?? "140"
);
export const RECOMMENDER_BUDGET_RETRIEVAL_MS = Number(
  process.env.RECOMMENDER_BUDGET_RETRIEVAL_MS ?? "260"
);
export const RECOMMENDER_BUDGET_RANKING_MS = Number(
  process.env.RECOMMENDER_BUDGET_RANKING_MS ?? "260"
);
export const RECOMMENDER_BUDGET_EXPLAINABILITY_MS = Number(
  process.env.RECOMMENDER_BUDGET_EXPLAINABILITY_MS ?? "300"
);
export const RECOMMENDER_BUDGET_SERIALIZE_MS = Number(
  process.env.RECOMMENDER_BUDGET_SERIALIZE_MS ?? "80"
);
export const RECOMMENDER_BUDGET_BUFFER_MS = Number(
  process.env.RECOMMENDER_BUDGET_BUFFER_MS ?? "100"
);
export const RECOMMENDER_BREAKER_MIN_REQUESTS = Number(
  process.env.RECOMMENDER_BREAKER_MIN_REQUESTS ?? "20"
);
export const RECOMMENDER_BREAKER_ERROR_RATE = Number(
  process.env.RECOMMENDER_BREAKER_ERROR_RATE ?? "0.6"
);
export const RECOMMENDER_BREAKER_CONSECUTIVE_FAILURES = Number(
  process.env.RECOMMENDER_BREAKER_CONSECUTIVE_FAILURES ?? "5"
);
export const RECOMMENDER_BREAKER_WINDOW_MS = Number(
  process.env.RECOMMENDER_BREAKER_WINDOW_MS ?? "60000"
);
export const RECOMMENDER_BREAKER_OPEN_MS = Number(
  process.env.RECOMMENDER_BREAKER_OPEN_MS ?? "30000"
);
export const RECOMMENDER_BREAKER_HALF_OPEN_PROBES = Number(
  process.env.RECOMMENDER_BREAKER_HALF_OPEN_PROBES ?? "3"
);
export const RECOMMENDER_BREAKER_HALF_OPEN_SUCCESS = Number(
  process.env.RECOMMENDER_BREAKER_HALF_OPEN_SUCCESS ?? "2"
);
export const RECOMMENDER_FEEDBACK_ENABLED =
  process.env.RECOMMENDER_FEEDBACK_ENABLED?.trim().toLowerCase() !== "false";
export const RECOMMENDER_FEEDBACK_DB_URL =
  process.env.RECOMMENDER_FEEDBACK_DB_URL?.trim() || process.env.DATABASE_URL?.trim() || "";
export const RECOMMENDER_EXPERIMENT_DEFAULT_ID =
  process.env.RECOMMENDER_EXPERIMENT_DEFAULT_ID?.trim() || "rec_v2_default";
export const RECOMMENDER_EXPERIMENT_SALT =
  process.env.RECOMMENDER_EXPERIMENT_SALT?.trim() || "rec_v2_salt";
export const RECOMMENDER_EXPERIMENT_TREATMENT_RATIO = Number(
  process.env.RECOMMENDER_EXPERIMENT_TREATMENT_RATIO ?? "0.5"
);
export const RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID =
  process.env.RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID?.trim() || "";
export const RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID =
  process.env.RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID?.trim() || "";
