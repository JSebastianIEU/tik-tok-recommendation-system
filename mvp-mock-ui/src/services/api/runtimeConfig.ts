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

function isLocalhostHost(hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase();
  return (
    normalized === "localhost" ||
    normalized === "127.0.0.1" ||
    normalized === "::1"
  );
}

function getDefaultApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return DEFAULT_LOCAL_API_BASE_URL;
  }

  return isLocalhostHost(window.location.hostname)
    ? DEFAULT_LOCAL_API_BASE_URL
    : normalizeBaseUrl(window.location.origin);
}

const configuredBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL ?? "");

export const API_BASE_URL =
  configuredBaseUrl || getDefaultApiBaseUrl();

export const MOCK_ONLY_MODE =
  isTruthy(import.meta.env.VITE_USE_MOCK_ONLY) || isGitHubPagesHost();

export function buildApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}
