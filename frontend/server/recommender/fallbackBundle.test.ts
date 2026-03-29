import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { FallbackBundleStore } from "./fallbackBundle";

test("FallbackBundleStore loads objective bundle from disk", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "fallback-bundle-"));
  const bundlePath = path.join(tempDir, "engagement.json");
  await fs.writeFile(
    bundlePath,
    JSON.stringify({
      objective: "engagement",
      generated_at: "2026-03-27T00:00:00Z",
      version: "fallback_bundle.v1",
      items: [{ candidate_id: "v1", score: 0.75 }]
    }),
    "utf-8"
  );
  const store = new FallbackBundleStore(tempDir, 60000);
  const loaded = await store.get("community");
  assert.ok(loaded);
  assert.equal(loaded?.objective, "engagement");
  assert.equal(loaded?.items.length, 1);
  await fs.rm(tempDir, { recursive: true, force: true });
});
