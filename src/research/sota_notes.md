# State-of-the-Art Retrieval Methods for Short-Form Social Content

## Overview
This document compares retrieval approaches for finding relevant TikTok-style videos based on captions, hashtags, and user preferences.

## 1. TF-IDF (Term Frequency-Inverse Document Frequency)

### What it is
A statistical measure that evaluates word importance in a document relative to a collection of documents.

### How it works for TikTok
- Converts video captions/hashtags into numerical vectors
- Scores words higher if they're frequent in a video but rare overall
- Example: "viral" appears in many videos = low score; "sourdough" is rare = high score

### Pros
- Very fast (no ML model needed)
- Easy to understand and debug
- Works well for keyword matching
- No training data required

### Cons
- No semantic understanding ("car" does not equal "automobile")
- Struggles with typos and slang
- Cannot capture context or intent

### Use case
Quick baseline for exact keyword matching (e.g., hashtag search)

## 2. BM25 (Best Match 25)

### What it is
An improved probabilistic ranking function, industry standard for search engines (used in Elasticsearch, Lucene).

### How it works
Enhanced TF-IDF with better handling of:
- Document length normalization
- Term saturation (diminishing returns for repeated words)
- Tunable parameters (k1, b)

### Pros
- Better than TF-IDF for short text
- Still very fast (under 10ms per query)
- Explainable results (keyword-based)
- Battle-tested in production systems

### Cons
- Still keyword-based (no semantic understanding)
- Requires parameter tuning for optimal results

### Use case
Recommended baseline for initial implementation

## 3. Sentence-BERT Embeddings

### What it is
Neural network that converts text into dense vector representations that capture semantic meaning.

### How it works
- Pre-trained transformer model (BERT-based)
- Maps sentences to 384/768-dimensional vectors
- Similar meanings produce similar vectors (cosine similarity)
- Example: "I love cooking" is similar to "Cooking is my passion"

### Pros
- Understands context and semantics
- Handles synonyms, paraphrasing, typos
- State-of-the-art quality for semantic search
- Works across languages (multilingual models available)

### Cons
- Slower inference (requires GPU for scale)
- "Black box" - hard to explain why things matched
- Requires more infrastructure (vector storage)
- Memory-intensive for large collections

### Use case
High-quality semantic retrieval after baseline is working

## 4. Hybrid Retrieval

### What it is
Combining keyword-based (BM25) and semantic (embeddings) retrieval for best of both worlds.

### How it works
```
User Query
    |
    v
[BM25 Stage] -> Top 1000 candidates (fast keyword filter)
    |
    v
[Embedding Stage] -> Re-rank top 100 (semantic quality)
    |
    v
Final Results
```

### Pros
- Balances speed and quality
- BM25 catches exact matches, embeddings catch semantic matches
- Can tune the blend (e.g., 70% BM25 + 30% embeddings)
- Production-grade approach used by modern search engines

### Cons
- More complex implementation
- Two systems to maintain
- Requires tuning the combination weights

### Use case
Production system after validating both components separately

## 5. FAISS Indexing (Facebook AI Similarity Search)

### What it is
Library for efficient similarity search and clustering of dense vectors at scale.

### How it works
- Builds optimized index structures (IVF, HNSW, etc.)
- Approximate nearest neighbor search (trades small accuracy for massive speed)
- Can search 100M+ vectors in milliseconds

### Pros
- Makes embedding search production-ready
- Highly optimized (C++ backend)
- Supports GPU acceleration
- Multiple index types for different tradeoffs

### Cons
- Only works with vector embeddings
- Adds infrastructure complexity
- Approximate search (may miss some results)

### Use case
Scaling layer for embedding-based retrieval (Sprint 3+)

## Comparison Table

| Method | Speed | Quality | Explainability | Complexity | Best For |
|--------|-------|---------|----------------|------------|----------|
| TF-IDF | Very Fast | Moderate | High | Low | Simple keyword search |
| BM25 | Very Fast | Good | High | Low | Initial baseline |
| Sentence-BERT | Slow | Excellent | Low | Medium | Semantic search |
| Hybrid | Fast | Very Good | Medium | High | Production system |
| FAISS | Very Fast | N/A | N/A | Medium | Scaling embeddings |

## Recommendation: Start with BM25

### Why BM25 as baseline?

1. Speed: Sub-10ms queries, no GPU required
2. Quality: Proven performance on short-form text (better than TF-IDF)
3. Explainability: Engineers and users can see keyword matches
4. Low Risk: Easy to implement, well-documented libraries (rank-bm25, Elasticsearch)
5. Migration Path: Can easily add embeddings later without discarding BM25

### Implementation Path

Sprint 2: BM25 baseline
- Index video captions and hashtags
- Basic retrieval for "find similar videos"
- A/B test against random recommendations

Sprint 3: Add Sentence-BERT
- Generate embeddings for all videos
- Compare BM25 vs embedding quality
- Measure latency impact

Sprint 4: Hybrid System
- Combine BM25 (fast filter) + embeddings (quality)
- Tune weights based on metrics
- Add FAISS if over 100k videos

### Academic Justification

Our choice is supported by:
1. Industry Standard: Used by Elasticsearch, AWS OpenSearch, Algolia
2. Research-Backed: BM25 outperforms TF-IDF in TREC benchmarks for short text (Robertson et al. 1994)
3. Pragmatism: Premature optimization with embeddings adds complexity without proven need

