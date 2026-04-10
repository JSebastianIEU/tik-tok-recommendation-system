import assert from "node:assert/strict";
import test, { afterEach } from "node:test";

process.env.RECOMMENDER_ENABLED = "true";
process.env.RECOMMENDER_BASE_URL = "http://recommender.test";
process.env.RECOMMENDER_RETRY_COUNT = "0";
process.env.RECOMMENDER_TIMEOUT_MS = "1000";

const { requestRecommendations, requestFabricSignals, requestCompatibility } = await import("./client");

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
  routing: {
    objective_requested: "community",
    objective_effective: "engagement",
    track: "post_publication"
  },
  portfolio: {
    enabled: true,
    weights: {
      reach: 0.45,
      conversion: 0.35,
      durability: 0.2
    },
    risk_aversion: 0.1,
    candidate_pool_cap: 120
  },
  top_k: 5,
  retrieve_k: 20
} as const;

test("requestRecommendations returns payload on 200", async () => {
  let calledUrl = "";
  let calledBody = "";
  globalThis.fetch = (async (input: URL | RequestInfo, init?: RequestInit) => {
    calledUrl = String(input);
    assert.equal(init?.method, "POST");
    calledBody = String(init?.body ?? "");
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
  assert.match(calledBody, /\"routing\"/);
  assert.match(calledBody, /\"portfolio\"/);
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

test("requestFabricSignals returns payload on 200", async () => {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        fabric_version: "fabric.v2",
        generated_at: "2026-03-22T00:00:00Z",
        video_id: "v1",
        as_of_time: "2026-03-22T00:00:00Z",
        registry_signature: "sig",
        extractor_traces: [],
        text: {
          token_count: 10,
          unique_token_count: 8,
          hashtag_count: 2,
          keyphrase_count: 3,
          cta_keyword_count: 1,
          clarity_score: 0.7,
          confidence: { raw: 0.8, calibrated: 0.76, calibration_version: "text-cal.v1" },
          missing: {}
        },
        structure: {
          hook_timing_seconds: 1.2,
          payoff_timing_seconds: 24,
          step_density: 0.6,
          pacing_score: 0.7,
          confidence: { raw: 0.8, calibrated: 0.76, calibration_version: "structure-cal.v1" },
          missing: {}
        },
        audio: {
          speech_ratio: null,
          tempo_bpm: null,
          energy: null,
          music_presence_score: null,
          confidence: { raw: 0.3, calibrated: 0.32, calibration_version: "audio-cal.v1" },
          missing: {}
        },
        visual: {
          shot_change_rate: null,
          visual_motion_score: null,
          style_tags: [],
          confidence: { raw: 0.3, calibrated: 0.32, calibration_version: "visual-cal.v1" },
          missing: {}
        },
        trace_ids: []
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    )) as typeof fetch;

  const result = await requestFabricSignals({
    video_id: "v1",
    as_of_time: "2026-03-22T00:00:00Z",
    caption: "abc"
  });
  assert.equal(result.ok, true);
});

test("requestCompatibility returns payload on 200", async () => {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        ok: true,
        bundle_id: "latest",
        fingerprints: { component: "recommender-learning-v1" },
        mismatches: []
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    )) as typeof fetch;
  const result = await requestCompatibility();
  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.payload.ok, true);
});
