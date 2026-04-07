import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { LabelingSessionStore } from "./store";

test("LabelingSessionStore creates sessions from benchmark sources and persists reviews", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "labeling-store-"));
  const benchmarksDir = path.join(tempRoot, "benchmarks");
  const sessionsDir = path.join(tempRoot, "sessions");
  await mkdir(benchmarksDir, { recursive: true });

  const benchmarkPath = path.join(benchmarksDir, "training_seed.json");
  await writeFile(
    benchmarkPath,
    JSON.stringify(
      {
        version: "recommender.human_comparable_benchmark.v1",
        generated_at: "2026-04-06T00:00:00.000Z",
        bundle_dir: "/tmp/bundle",
        sample_metadata: {},
        rubric: {},
        cases: [
          {
            case_id: "engagement::q1",
            objective: "engagement",
            query: {
              query_id: "q1",
              display: { caption: "query caption" },
              query_payload: { description: "query caption" }
            },
            retrieve_k: 10,
            label_pool_size: 2,
            source_candidate_pool_size: 8,
            candidates: [
              {
                candidate_id: "c1",
                display: { caption: "candidate 1", video_url: "https://www.tiktok.com/@a/video/1" },
                candidate_payload: { video_url: "https://www.tiktok.com/@a/video/1" },
                baseline_rank: 1,
                baseline_score: 0.92,
                support_level: "full",
                ranking_reasons: ["strong_semantic_relevance"],
                label: "good"
              },
              {
                candidate_id: "c2",
                display: { caption: "candidate 2" },
                candidate_payload: {},
                baseline_rank: 2,
                baseline_score: 0.71,
                support_level: "partial",
                ranking_reasons: ["strong_intent_alignment"],
                label: "bad"
              }
            ]
          }
        ]
      },
      null,
      2
    ),
    "utf-8"
  );

  const store = new LabelingSessionStore({
    benchmarksDir,
    sessionsDir
  });

  const sources = await store.listSources();
  assert.equal(sources.length, 1);
  assert.equal(sources[0]?.source_id, "training_seed.json");

  const session = await store.createSession({ sourceId: "training_seed.json" });
  assert.equal(session.cases.length, 1);
  assert.equal(session.summary.reviewed_count, 0);
  assert.equal(session.cases[0]?.candidates[0]?.review.label, null);

  const updated = await store.updateCandidateReview({
    sessionId: session.session_id,
    caseId: "engagement::q1",
    candidateId: "c1",
    label: "saved"
  });
  assert.equal(updated.summary.reviewed_count, 1);
  assert.equal(updated.summary.saved_count, 1);
  assert.equal(updated.cases[0]?.candidates[0]?.review.label, "saved");

  const sessions = await store.listSessions();
  assert.equal(sessions.length, 1);
  assert.equal(sessions[0]?.summary.saved_count, 1);

  const persisted = JSON.parse(
    await readFile(path.join(sessionsDir, `${session.session_id}.json`), "utf-8")
  ) as {
    cases: Array<{ candidates: Array<{ review: { label: string | null } }> }>;
  };
  assert.equal(persisted.cases[0]?.candidates[0]?.review.label, "saved");

  await rm(tempRoot, { recursive: true, force: true });
});
