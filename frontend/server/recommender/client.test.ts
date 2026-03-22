import assert from "node:assert/strict";
import test, { afterEach } from "node:test";

process.env.RECOMMENDER_ENABLED = "true";
process.env.RECOMMENDER_BASE_URL = "http://recommender.test";
process.env.RECOMMENDER_RETRY_COUNT = "0";
process.env.RECOMMENDER_TIMEOUT_MS = "1000";

const { requestRecommendations } = await import("./client");

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

const payload = {
  objective: "engagement",
  as_of_time: "2026-03-22T00:00:00Z",
  query: {
    query_id: "q1",
    text: "growth tutorial"
  },
  candidates: [
    {
      candidate_id: "c1",
      text: "growth tips",
      as_of_time: "2026-03-20T00:00:00Z"
    }
  ],
  top_k: 5,
  retrieve_k: 20
} as const;

test("requestRecommendations returns payload on 200", async () => {
  let calledUrl = "";
  globalThis.fetch = (async (input: URL | RequestInfo, init?: RequestInit) => {
    calledUrl = String(input);
    assert.equal(init?.method, "POST");
    return new Response(
      JSON.stringify({
        objective: "engagement",
        objective_effective: "engagement",
        generated_at: "2026-03-22T00:00:00Z",
        fallback_mode: false,
        items: [
          {
            candidate_id: "c1",
            rank: 1,
            score: 0.91,
            similarity: { sparse: 0.8, dense: 0.9, fused: 0.85 }
          }
        ]
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }) as typeof fetch;

  const result = await requestRecommendations(payload);
  assert.equal(calledUrl, "http://recommender.test/v1/recommendations");
  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.payload.items.length, 1);
  assert.equal(result.payload.items[0]?.candidate_id, "c1");
});

test("requestRecommendations returns failure payload on non-2xx", async () => {
  globalThis.fetch = (async () => new Response("upstream failed", { status: 503 })) as typeof fetch;
  const result = await requestRecommendations(payload);
  assert.equal(result.ok, false);
  if (result.ok) {
    return;
  }
  assert.equal(result.status, 503);
  assert.match(result.error, /upstream failed/i);
});

test("requestRecommendations returns failure payload on fetch exception", async () => {
  globalThis.fetch = (async () => {
    throw new Error("network down");
  }) as typeof fetch;
  const result = await requestRecommendations(payload);
  assert.equal(result.ok, false);
  if (result.ok) {
    return;
  }
  assert.match(result.error, /network down/i);
});
