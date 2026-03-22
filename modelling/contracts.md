# Modelling Contracts

## Data Contract Layer (`contract.v1`)

Implemented in:

- `src/recommendation/contracts.py`

Canonical entities:

- `CanonicalAuthor`
- `CanonicalVideo`
- `CanonicalVideoSnapshot`
- `CanonicalComment`
- `CanonicalCommentSnapshot`

Governance checks:

- runtime schema validation (pydantic)
- referential integrity + uniqueness checks
- point-in-time (`as_of_time`) checks
- feature access policy by modeling track (`pre_publication`, `post_publication`)
- time-series ingest support for repeated `video_id` rows
- snapshot timestamp precedence with optional strict fallback policy

## CandidateProfileCore (`core.v1`)

Built from upload-time inputs:

- description
- hashtags
- mentions
- objective
- audience
- content_type
- primary_cta
- locale

Key outputs:

- normalized text and tags
- token blocks and keyphrases
- topic/format tags
- retrieval text
- quality confidence + flags

## CandidateSignalProfile (`extractors.v1`)

Built from `CandidateProfileCore` + optional media hints:

- duration/transcript/OCR hints
- scene cut / motion / tempo / speech hints

Key outputs:

- visual signals
- audio signals
- transcript/OCR signals
- structure signals
- overall confidence + quality flags

## ComparableNeighborhood (`step2.v1`)

Built from:

- `CandidateProfileCore`
- `CandidateSignalProfile` (optional)
- historical candidate records

Output blocks:

- `content_twins[]`
- `similar_overperformers[]`
- `similar_underperformers[]`
- `ranking_traces[]`
- `confidence`

Scoring model:

`composite = 0.45*text_similarity + 0.20*hashtag_similarity + 0.15*intent_match + 0.10*format_match + 0.10*signal_match`

Selection constraints:

- stage-1 pool size: 40
- max per author in each neighborhood: 2
- max per topic in each neighborhood: 4
- residual split uses Q1/Q3 bands on `log1p(views) - expected_log_views`

## NeighborhoodContrast (`step3.v1`)

Built from:

- `ComparableNeighborhood`
- `CandidateProfileCore`
- `CandidateSignalProfile` (optional)

Output blocks:

- `normalized_deltas[]` (feature-level over vs under contrast)
- `claims[]` (evidence-backed actionable claims)
- `conflicts[]` (contradictory signal flags)
- `summary` (top strengths and risks)
- `fallback_mode` + `neighborhood_confidence`

Claim gating rules:

- minimum support in both groups
- minimum absolute normalized effect (`|z_delta|`)
- minimum reliability threshold

Claim evidence is deterministic:

- no hardcoded per-topic heuristics for individual videos
- no LLM required for claim generation
- each claim includes traceability to feature keys and candidate keys

## TrainingDataMart (`datamart.v1`)

Built from:

- `CanonicalDatasetBundle` (`contract.v1`)

Python implementation:

- `src/recommendation/datamart.py`

Primary outputs:

- `rows[]` (`TrainingRow`) with:
  - leakage-safe `as_of_time`
  - pre-window features + missingness flags
  - post-window labels (reach/engagement/conversion)
  - author-baseline adjusted residual labels
  - deterministic split assignment (`train` / `validation` / `test`)
- `pair_rows[]` (`PairTrainingRow`) for retrieval/ranking training
- `excluded_video_records[]` with structured censoring reasons

Core safeguards:

- right-censoring when label horizon is unavailable
- strict point-in-time policy checks against contract bundle timestamps
- track-aware feature policy (`pre_publication` excludes comments)
- reproducible split logic by `as_of_time` ordering
- train-only target normalization statistics
- train-safe author baselines (leave-one-out in train; train-only references for non-train rows)

## Non-Goals

- No LLM requirement in feature extraction.
- No LLM requirement in contrast/claim generation.
- No direct ranking in this layer.
- No post-publication metrics leakage into candidate representation.
