# Research Documentation

> Status note: this directory captures earlier retrieval-focused experiments and notes (standalone scripts such as `run_experiment.py` on mock JSONL).
> It is useful background context, but it is not the best guide to the current recommender runtime in `src/recommendation/` (production retrieval is `learning/retriever.py`, not the removed `src/retrieval/` scaffold).

## Purpose
This directory contains research and experimental validation for retrieval methods used in earlier stages of the TikTok-style recommendation system.

## Contents

### sota_notes.md
State of the art comparison of retrieval approaches:
- TF-IDF
- BM25
- Sentence-BERT embeddings
- Hybrid retrieval
- FAISS indexing

Includes technical explanations, pros/cons, use cases, and academic references.

### experiments.md
Experimental validation comparing BM25 and TF-IDF on actual mock data from the project. Results show 75% precision with BM25 as the recommended baseline.

Key findings:
- BM25 is faster than TF-IDF (0.00ms vs 1.63ms average)
- Both methods achieved 75% precision on test queries
- Vocabulary mismatch is the primary failure case (25% of queries)
- Semantic search needed for Sprint 3 to address gaps

### run_experiment.py
Python script used to run the retrieval experiments. Can be re-run with updated data or queries.

## Historical Recommendation

**Use BM25 as the Sprint 2 baseline**

Justification:
1. Speed: Sub-millisecond query latency, no GPU required
2. Quality: 75% precision on test queries with well tagged content
3. Explainability: Clear keyword matching, easy to debug
4. Low risk: Simple implementation with well documented libraries
5. Scalable: Can add semantic search in Sprint 3 without discarding BM25

## Implementation Roadmap

### Sprint 2: BM25 Baseline
- Index video captions, hashtags, and keywords
- Implement retrieval endpoint
- AB test against random recommendations
- Track failed queries for future improvement

### Sprint 3: Semantic Layer
- Implement Sentence-BERT for queries that fail BM25
- Compare hybrid (BM25 + embeddings) vs BM25 only
- Measure latency and quality tradeoffs

### Sprint 4: Production Optimization
- Hybrid retrieval (BM25 first pass, embeddings rerank)
- Add FAISS indexing for scale
- Fine-tune based on user engagement metrics

## Running the Experiment

Requirements:
```bash
pip install rank-bm25 scikit-learn
```

Run:
```bash
python src/research/run_experiment.py
```

Output shows comparative results for BM25 vs TF-IDF on mock data.

For the current production-oriented architecture, refer to `src/recommendation/README.md` and [docs/architecture/recommender_overview.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/architecture/recommender_overview.md).

## References

- Robertson et al. (1994) - "Okapi at TREC-3" (BM25 algorithm)
- Reimers & Gurevych (2019) - "Sentence-BERT" (EMNLP)
- Johnson et al. (2019) - "Billion-scale similarity search with FAISS"
