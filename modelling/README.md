# Modelling Workspace

This folder tracks modelling contracts and design notes that sit alongside the implemented recommendation platform.

> Status note: this is a supporting design workspace, not the primary source of truth for the live runtime.
> For the current implemented architecture, start with [docs/architecture/recommender_overview.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/architecture/recommender_overview.md) and `src/recommendation/`.

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

- `src/recommendation`: current implementation for contracts, datamart, fabric, comment intelligence, learning, runtime inference, and service boundaries.
- `frontend/server/`: current Node gateway for product-facing report generation, recommender proxying, fallback handling, and feedback capture.
- `modelling/`: supporting design references and evolution notes.

Treat `src/recommendation` and `frontend/server/` as the source of truth for behavior. Treat this folder as design context.

## Modeling Progress

- Data Contract Layer implemented in Python (`contract.v2`) with runtime schemas, bitemporal policy checks, watermarking, and manifest replay support.
- Comment Intelligence Layer implemented in Python (`comment_intelligence.v2`) with deterministic taxonomy/intent features, sentiment dynamics, reply-graph metrics, and transfer priors for pre-publication flows.
- Step 1 implemented and tested.
- Part 2 implemented and tested.
- Step 2 implemented with composite scoring, residual-based bands, author/topic caps, and ranking traces.
- Step 3 implemented with normalized deltas, reliability weighting, conflict flags, and claim/action generation.
- Training Data Mart implemented in Python with leakage-safe windows, censoring, author-baseline residualization, and time-based splits.
