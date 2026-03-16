# Retrieval Module

This module provides a baseline text retrieval system for TikTok posts using **TF-IDF cosine similarity**. It's designed to be modular and easy to upgrade with more advanced algorithms.

## Overview

The retrieval system implements a simple pipeline:
1. **Indexing**: Load posts, combine textual fields, build TF-IDF vectors
2. **Searching**: Transform query, compute similarity scores, return top-k results

### What it does
- Combines `caption`, `hashtags`, `keywords`, and `search_query` fields into a single searchable document per post
- Uses scikit-learn's TfidfVectorizer for fast baseline retrieval
- Returns top-k results with `{video_id, video_url, score, caption, hashtags}`

### What it doesn't do (yet)
- Advanced ranking (BM25, semantic embeddings)
- Personalization or user context
- Real-time updates
- Multi-modal search (video content, audio)

## File Structure

```
src/retrieval/
├── index.py          # RetrievalIndex class: builds and persists TF-IDF index
├── search.py         # search() function: queries the index, returns top-k
└── README.md         # This file

scripts/
├── build_index.py    # CLI: build index from mock data
└── query_index.py    # CLI: query the index with text
```

## Usage

### 1. Build the index

```bash
python scripts/build_index.py
```

This loads `data/mock/tiktok_posts_mock.jsonl` and saves a pickled index to `data/mock/retrieval_index.pkl`.

**Output:**
```
Loading posts from .../data/mock/tiktok_posts_mock.jsonl...
Loaded 50 posts
Building TF-IDF index...
Built index with 50 posts and 217 features
Saving index to .../data/mock/retrieval_index.pkl...
Saved index to .../data/mock/retrieval_index.pkl

✓ Index built successfully!
  Posts indexed: 50
  Index location: .../data/mock/retrieval_index.pkl
```

### 2. Query the index

```bash
python scripts/query_index.py "core workout"
```

**Output:**
```
Query: 'core workout'
Top 10 results:

1. v001 (score: 0.7234)
   Caption: 5-minute core burner
   Hashtags: #fitness, #core, #workout
   URL: https://www.tiktok.com/@fitjules/video/700000001

2. v007 (score: 0.4521)
   Caption: 10-min evening stretch
   Hashtags: #yoga, #stretch, #wellness
   URL: https://www.tiktok.com/@yogawithmei/video/700000007
...
```

**Options:**
- `--topk N`: Return top N results (default: 10)
- `--language <code>`: Filter by language (e.g., `en`)
- `--min-likes N`: Filter by likes threshold
- `--min-score F`: Drop low-relevance results below score threshold (default: `0.0`)
- `--json`: Output results as JSON

```bash
python scripts/query_index.py "ramen recipe" --topk 3 --json
```

## Programmatic Usage

```python
from src.retrieval.index import RetrievalIndex
from src.retrieval.search import search

# Load pre-built index
index = RetrievalIndex.load("data/mock/retrieval_index.pkl")

# Search
results = search(index, query="yoga tips", topk=5)

# Results format:
# [
#   {
#     "video_id": "v007",
#     "video_url": "https://...",
#     "score": 0.8234,
#     "caption": "10-min evening stretch",
#     "hashtags": ["#yoga", "#stretch", "#wellness"],
#     "likes": 12345,
#     "language": "en"
#   },
#   ...
# ]
```

## Architecture & Modularity

The system is designed for **easy algorithm swapping**. Future improvements can be added by:

### Option 1: Extend `RetrievalIndex`
Add alternative index types (e.g., `BM25Index`, `EmbeddingIndex`) that inherit from a base interface.

### Option 2: Plugin pattern in `search()`
Pass a `backend` parameter to switch algorithms:
```python
search(index, query, backend="bm25")  # TODO
search(index, query, backend="sbert")  # TODO
```

### Option 3: Separate ranking modules
Keep indexing generic, implement ranking in separate modules:
```python
from src.ranking.bm25 import rank_bm25  # TODO
from src.ranking.sbert import rank_semantic  # TODO
```

## Future Enhancements (TODOs)

### Ranking Algorithms
- **BM25**: Better term frequency handling (via `rank-bm25` library)
- **SBERT**: Semantic embeddings with sentence-transformers
- **FAISS**: Dense vector search for large-scale retrieval
- **Hybrid**: Combine lexical (TF-IDF/BM25) + semantic (SBERT) signals

### Features
- Query expansion (synonyms, related terms)
- Spelling correction
- Multi-lingual support (language-specific stemmers, stop words)
- Filtering (date range, author, language, minimum likes)
- Personalized ranking (user history, preferences)
- Ranking explanations (which terms matched)

### Scalability
- Incremental index updates (add new posts without full rebuild)
- Distributed indexing for large datasets
- Index compression and memory optimization
- Cloud storage integration (S3, GCS)

### Quality
- Evaluation metrics (NDCG, MRR, Recall@K)
- A/B testing framework for comparing algorithms
- Query logs and analytics
- Index quality metrics (vocabulary size, sparsity)

## Notes for Omar (Ranking Owner)

This baseline is ready for experimentation. Here's how to plug in different algorithms:

1. **BM25**: Replace `TfidfVectorizer` with `rank_bm25.BM25Okapi` in [index.py:53-60](src/retrieval/index.py#L53-L60)
2. **SBERT**: Use `sentence-transformers` to encode texts, store embeddings in [index.py:19](src/retrieval/index.py#L19)
3. **FAISS**: Build FAISS index from embeddings, update search logic in [search.py:24-30](src/retrieval/search.py#L24-L30)

All TODO comments in the code mark extension points. The current implementation handles:
- ✅ End-to-end pipeline on mock data
- ✅ Modular design for algorithm swaps
- ✅ Clean output format
- ✅ Persistence (save/load index)
