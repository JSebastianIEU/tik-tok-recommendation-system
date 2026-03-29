import assert from "node:assert/strict";
import test from "node:test";

import { createUuidV7 } from "./requestId";

test("createUuidV7 returns UUIDv7-like formatted id", () => {
  const id = createUuidV7(1735689600000);
  assert.match(
    id,
    /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
  );
});

