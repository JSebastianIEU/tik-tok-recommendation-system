# Recommender Data Governance

## Principles

- Derived-only at rest for new recommender outputs.
- No raw UGC text persistence in new trace or report artifacts.
- Store IDs, hashes, numeric scores, and compact traces only.
- Prefer deterministic replay over ad hoc manual edits.

## Allowed Stored Fields

- Request ids, candidate ids, bundle ids, and manifest ids.
- Numeric metrics, calibration scores, trajectory scores, graph scores, alignment scores, and portfolio traces.
- Missingness flags and typed fallback reasons.
- Control-plane summaries and drift outputs.

## Disallowed Stored Fields

- Raw comment text beyond transient in-memory processing.
- Raw transcript/OCR content in derived outputs.
- Full prompt bodies or secrets.
- Unmasked user-identifiable payloads that are not required for debugging.

## Retention

- Keep derived validation artifacts for operational debugging only.
- Archive or prune intermediate artifacts on a fixed schedule once the bundle is promoted or superseded.
- Keep incident reports and signed-off release evidence according to team policy.

## Access Control

- Restrict direct DB access to service owners and on-call responders.
- Use the minimum needed permissions for feedback persistence and control-plane jobs.
- Treat compatibility, drift, and feedback rows as operational telemetry, not product analytics exports.

## Audit Rules

- Every new artifact path must remain derived-only.
- Every new report must state its scope, time window, and fallback behavior.
- Any addition that stores raw text must be justified explicitly before merge.

## Operational Checks

- Verify that `feedback_store` only contains derived traces.
- Verify that `rec_request_events`, `rec_candidate_events`, and `rec_served_outputs` remain limited to request and candidate metadata.
- Verify that control-plane reports do not expand to raw UGC capture.
