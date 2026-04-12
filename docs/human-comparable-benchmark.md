# Human Comparable Benchmark

This benchmark is the Track 1 bootstrap path for the Phase 2 learned reranker.

It gives us a small, reusable set of real query videos with a fixed comparable pool
that a human can label as:

- `good`
- `unclear`
- `bad`

## Why this exists

We do not yet have enough explicit live comparable feedback to train the learned reranker
purely from `rec_ui_feedback_events`.

So the benchmark solves two immediate problems:

1. it gives us a trustworthy offline evaluation set
2. it gives us a seed supervision source for manual pair construction (`good > bad`)

## Workflow

### 1. Build the benchmark pack

Generate a benchmark JSON from real DB-backed videos and the current recommender bundle:

```bash
DATABASE_URL=... python3 scripts/build_human_comparable_benchmark.py \
  --db-url "$DATABASE_URL" \
  --bundle-dir artifacts/recommender_real/latest \
  --output-path artifacts/benchmarks/human_comparable_benchmark.json \
  --objectives engagement \
  --query-count 24 \
  --label-pool-size 12
```

The output contains:

- top-level rubric/instructions
- real query payloads
- a fixed candidate pool per query
- frozen baseline ranks/scores for those candidates
- explicit `label: null` and `label_notes: ""` fields for human annotation

### 1b. Build a disposable training source for UI labeling

For training-label collection, keep the held-out benchmark separate and generate a
second source file, for example:

```bash
DATABASE_URL=... python3 scripts/build_human_comparable_benchmark.py \
  --db-url "$DATABASE_URL" \
  --bundle-dir artifacts/recommender_real/latest \
  --output-path artifacts/benchmarks/human_comparable_training_seed.json \
  --objectives engagement \
  --query-count 24 \
  --label-pool-size 10
```

That training seed is only a frozen source pack. The click-based review labels are
stored separately in disposable session files under:

- `artifacts/labeling_sessions`

So the source pack can be regenerated or deleted later without losing the held-out
benchmark.

### 2. Label the cases

For each candidate in each case:

- `good`: strong historical comparable for the query/objective
- `unclear`: mixed or ambiguous comparable
- `bad`: poor comparable

Edit each candidate object directly:

```json
{
  "candidate_id": "...",
  "display": { "...": "..." },
  "candidate_payload": { "...": "..." },
  "baseline_rank": 1,
  "baseline_score": 0.52,
  "support_level": "partial",
  "ranking_reasons": ["strong_intent_alignment"],
  "label": "good",
  "label_notes": "Topically close and useful as a reference example."
}
```

Use `unclear` when the candidate is not confidently good or bad.

### 2b. Label training cases in the local UI

For the training-label workflow, use the local UI at:

- `/labeling`

That screen creates a disposable review session from a benchmark-style source pack
and stores per-candidate labels as:

- `saved`
- `relevant`
- `not_relevant`

Those labels live only in the session JSON files under `artifacts/labeling_sessions`.
They do not overwrite the original benchmark source file.

### 3. Evaluate a bundle against the labeled benchmark

Once labels are filled in:

```bash
python3 scripts/eval_human_comparable_benchmark.py \
  --bundle-dir artifacts/recommender_real/latest \
  --benchmark-json artifacts/benchmarks/human_comparable_benchmark.json \
  --output-path artifacts/benchmarks/human_comparable_benchmark_eval.json
```

This reruns the current bundle against the fixed candidate pool and reports:

- `ndcg@k`
- `mrr@k`
- `recall@k`
- `good_rate@k`, `unclear_rate@k`, `bad_rate@k`
- baseline-snapshot metrics from the frozen original ordering
- per-case learned-reranker metadata when present

Cases with no `good` labels are still included in the benchmark output:

- `ndcg@k` still uses the graded `good > unclear > bad` relevance map
- `good_rate@k`, `unclear_rate@k`, and `bad_rate@k` still report label composition at the top of the ranking
- `mrr@k` and `recall@k` are omitted for those cases because there is no positive target to retrieve

This is intentional, because “no good comparable exists in this candidate pool” is an important evaluation signal, not a case we want to silently drop.

## Labeling policy

The benchmark rubric intentionally uses a simple 3-way scale:

- `good` -> strong positive
- `unclear` -> graded but ignored for pairwise training
- `bad` -> strong negative

For pairwise bootstrap training, only:

- `good > bad`

should be used initially.

`unclear` should be excluded from pairwise bootstrap rows until we have a more mature policy.
