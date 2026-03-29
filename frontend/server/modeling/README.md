# Modeling

This folder contains the recommendation modeling pipeline interfaces used by the local API.

## Python-Owned Components

Data Contract Layer (`contract.v2`) and Training Data Mart (`datamart.v1`) are owned by the
Python implementation in `src/recommendation` and are no longer implemented in this TypeScript
runtime.

Feature/Signal Fabric (`fabric.v2`) is also Python-owned. The TypeScript Part 2 extractor
remains as a temporary shim/fallback path while parity checks are completed.

## Step 1: Candidate Representation (implemented)

`step1/buildCandidateProfileCore.ts` builds a deterministic `CandidateProfileCore` from
upload-time inputs:

- description
- hashtags
- mentions
- creator intent fields (objective, audience, content type, CTA, locale)

The output is versioned (`core.v1`) and designed to be stable enough for retrieval and
downstream analytics without requiring an LLM.

## Part 2: Signal Extractors (implemented, deterministic)

`part2/extractCandidateSignals.ts` computes deterministic feature blocks from:

- core candidate profile
- optional media hints supplied by the client/server

It outputs:

- visual features
- audio features
- transcript/OCR features
- structure features
plus quality/confidence metadata for downstream recommendation logic.

## Step 2: Retrieval + Neighborhood Split (implemented, deterministic)

`step2/buildComparableNeighborhood.ts` builds a ranked comparable neighborhood from:

- candidate core profile
- candidate signal profile
- historical candidate pool

It performs:

- stage-1 candidate generation
- composite deterministic reranking
- residual-based overperformer/underperformer split
- author/topic diversity caps
- confidence and ranking trace generation

## Step 3: Contrast & Evidence Builder (implemented, deterministic)

`step3/buildNeighborhoodContrast.ts` builds structured contrast evidence from Step 2 neighborhoods:

- normalized feature deltas
- reliability-weighted evidence scoring
- claim gating with minimum support/effect thresholds
- conflict detection across competing signals
- claim-level confidence and evidence traces
