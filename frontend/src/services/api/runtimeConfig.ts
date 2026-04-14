const DEFAULT_LOCAL_API_BASE_URL = "http://localhost:5174";

function normalizeBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

function isTruthy(value: string | undefined): boolean {
  return value?.trim().toLowerCase() === "true";
}

function isGitHubPagesHost(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  const hostname = window.location.hostname.toLowerCase();
  return hostname.endsWith("github.io");
}

function isVercelHost(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  const hostname = window.location.hostname.toLowerCase();
  return hostname.endsWith(".vercel.app") || hostname.endsWith(".vercel.sh");
}

const configuredBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL ?? "");

// When running on Vercel, ignore any localhost API URL that was baked in at
// build time and use the deployment's own origin instead.
const isLocalhostUrl =
  configuredBaseUrl.includes("localhost") ||
  configuredBaseUrl.includes("127.0.0.1");

export const API_BASE_URL =
  configuredBaseUrl && !(isVercelHost() && isLocalhostUrl)
    ? configuredBaseUrl
    : isVercelHost()
      ? window.location.origin
      : DEFAULT_LOCAL_API_BASE_URL;

export const MOCK_ONLY_MODE =
  isTruthy(import.meta.env.VITE_USE_MOCK_ONLY) || isGitHubPagesHost();

export function buildApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}
