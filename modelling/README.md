# Modelling Workspace

This folder tracks recommendation architecture decisions and data contracts shared across the system.

## Current Scope

1. `Step 1` candidate representation contract (`CandidateProfileCore`)
2. `Part 2` deterministic signal extractor contract (`CandidateSignalProfile`)
3. `Step 2` retrieval and neighborhood contract (`ComparableNeighborhood`)
4. `Step 3` neighborhood contrast contract (`NeighborhoodContrast`)
5. `Component` training data mart contract (`TrainingDataMart`)
6. `Component` comment intelligence layer (`comment_intelligence.v2`)

## Design Principles

- Core intelligence is deterministic and reproducible.
- No LLM dependency for feature extraction or recommendation evidence.
- Candidate profiles are versioned to support safe evolution.
- Signal extraction includes confidence and quality flags so downstream logic can degrade gracefully.

## Runtime Location

- `frontend/server/modeling`: Step 1 core profile, Part 2 deterministic extractors, Step 2 retrieval/neighborhood, Step 3 contrast.
- `src/recommendation`: Data Contract Layer (`contract.v2`), Feature/Signal Fabric (`fabric.v2`), and Training Data Mart (`datamart.v1`).
- This folder remains the source of truth for product-facing modelling specs and evolution notes.

## Modeling Progress

- Data Contract Layer implemented in Python (`contract.v2`) with runtime schemas, bitemporal policy checks, watermarking, and manifest replay support.
- Comment Intelligence Layer implemented in Python (`comment_intelligence.v2`) with deterministic taxonomy/intent features, sentiment dynamics, reply-graph metrics, and transfer priors for pre-publication flows.
- Step 1 implemented and tested.
- Part 2 implemented and tested.
- Step 2 implemented with composite scoring, residual-based bands, author/topic caps, and ranking traces.
- Step 3 implemented with normalized deltas, reliability weighting, conflict flags, and claim/action generation.
- Training Data Mart implemented in Python with leakage-safe windows, censoring, author-baseline residualization, and time-based splits.
