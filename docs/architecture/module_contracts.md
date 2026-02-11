# Module Contracts and Interface Specifications

## Overview

This document defines the input/output contracts and interface specifications for each module in the TikTok Recommendation System. These contracts ensure consistent communication between modules and enable independent development and testing.

**Last Updated:** February 11, 2026  
**Sprint:** 1  
**Status:** Initial version

---

## 1. Data Layer Module

**Location:** `src/common/`  
**Owner:** Data Engineering Team  
**Purpose:** Canonical data schemas and validation logic

### 1.1 Schema Contract (`schemas.py`)

#### Core Types

**TikTokPost**
```python
{
    "video_id": str,                    # Unique identifier
    "video_url": HttpUrl,               # Valid TikTok URL
    "caption": str,                     # Video caption text
    "hashtags": List[str],              # List of hashtag strings
    "keywords": List[str],              # Extracted keywords
    "search_query": str,                # Original search query
    "posted_at": datetime,              # ISO 8601 timestamp
    "likes": int,                       # >= 0
    "comments_count": int,              # >= 0
    "shares": int,                      # >= 0
    "views": int,                       # >= 0
    "author": Author,                   # Author object
    "audio": Audio,                     # Audio object
    "video_meta": VideoMeta,            # Video metadata
    "comments": List[Comment]           # List of comments (min 3)
}
```

**Author**
```python
{
    "author_id": str,
    "username": str,
    "followers": int                    # >= 0
}
```

**Audio**
```python
{
    "audio_id": str,
    "audio_title": str
}
```

**VideoMeta**
```python
{
    "duration_seconds": int,            # >= 1
    "language": str                     # ISO language code
}
```

**Comment**
```python
{
    "comment_id": str,
    "text": str,
    "likes": int,                       # >= 0
    "created_at": datetime              # ISO 8601 timestamp
}
```

#### Validation Rules

- All numeric fields must be non-negative
- `comments` list must contain at least 3 items
- `duration_seconds` must be > 0
- URLs must be valid HTTP/HTTPS
- Datetimes must be parseable ISO 8601 format

### 1.2 Validation Contract (`validation.py`)

#### Function: `validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]`

**Input:**
- `record`: Dictionary representation of a TikTok post

**Output:**
- Tuple of (is_valid: bool, errors: List[str])
- If valid: `(True, [])`
- If invalid: `(False, ["error message 1", "error message 2"])`

**Validation Steps:**
1. Pydantic schema validation
2. Business rule validation (comment count, numeric constraints)
3. Datetime parsing validation

#### Function: `validate_jsonl(path: Path) -> Tuple[int, int, List[str]]`

**Input:**
- `path`: Path to JSONL file

**Output:**
- Tuple of (total_records: int, valid_records: int, all_errors: List[str])

---

## 2. Retrieval Module

**Location:** `src/retrieval/`  
**Owner:** ML Engineering Team  
**Purpose:** Index building and similarity search

### 2.1 Index Contract (`index.py`)

#### Class: `RetrievalIndex`

**Method: `build(posts: List[TikTokPost]) -> None`**

**Input:**
- `posts`: List of validated TikTokPost objects

**Output:**
- None (mutates internal state)

**Side Effects:**
- Builds TF-IDF vectorizer from combined text fields
- Creates sparse matrix representation
- Stores posts internally

**Text Combination Strategy:**
```python
combined_text = caption + " " + " ".join(hashtags) + " " + " ".join(keywords) + " " + search_query
```

---

**Method: `save(path: Path) -> None`**

**Input:**
- `path`: Output path for serialized index

**Output:**
- None (writes to disk)

**Format:**
- Pickle file containing:
  - `posts`: List[Dict] (serialized TikTokPost objects)
  - `vectorizer`: TfidfVectorizer instance
  - `matrix`: scipy sparse matrix

---

**Method: `load(path: Path) -> RetrievalIndex`** (classmethod)

**Input:**
- `path`: Path to serialized index file

**Output:**
- `RetrievalIndex` instance with loaded state

**Exceptions:**
- `FileNotFoundError` if path doesn't exist
- `ValueError` if file is corrupted

---

**Method: `get_posts() -> List[TikTokPost]`**

**Output:**
- List of all indexed posts

---

**Method: `get_vectorizer() -> Optional[TfidfVectorizer]`**

**Output:**
- Fitted vectorizer or None if not built

---

**Method: `get_matrix() -> Optional[csr_matrix]`**

**Output:**
- Sparse TF-IDF matrix or None if not built

---

### 2.2 Search Contract (`search.py`)

#### Function: `search(index: RetrievalIndex, query: str, topk: int = 10) -> List[Dict[str, Any]]`

**Input:**
- `index`: Pre-built RetrievalIndex instance
- `query`: Search query string
- `topk`: Number of results to return (default: 10)

**Output:**
- List of result dictionaries, sorted by relevance (descending)

**Result Format:**
```python
[
    {
        "video_id": str,
        "video_url": str,                # Serialized URL
        "score": float,                  # Cosine similarity [0, 1]
        "caption": str,
        "hashtags": List[str]
    },
    ...
]
```

**Behavior:**
- Returns empty list if index is not built
- Returns at most `topk` results
- Scores are cosine similarities (higher = more relevant)

---

#### Function: `filtered_search(index: RetrievalIndex, query: str, topk: int = 10, language: str = None, min_likes: int = None) -> List[Dict[str, Any]]`

**Input:**
- `index`: Pre-built RetrievalIndex
- `query`: Search query string
- `topk`: Number of results (default: 10)
- `language`: Optional language filter (ISO code)
- `min_likes`: Optional minimum likes threshold

**Output:**
- List of filtered result dictionaries (same format as `search()`)

**Filtering Logic:**
1. Over-fetch results (topk * 2)
2. Apply language filter (if specified)
3. Apply likes filter (if specified)
4. Return first topk results after filtering

---

## 3. Baseline Analytics Module

**Location:** `src/baseline/`  
**Owner:** Data Science Team  
**Purpose:** Statistical analysis and engagement metrics

### 3.1 Analytics Contract (`baseline_stats.py`)

#### Function: `compute_baseline_report(posts: List[TikTokPost]) -> Dict[str, Any]`

**Input:**
- `posts`: List of TikTokPost objects

**Output:**
- Dictionary containing analytics results

**Output Structure:**
```python
{
    "metadata": {
        "total_posts": int,
        "generated_at": str             # ISO timestamp
    },
    "descriptive_stats": {
        "views": {
            "count": int,
            "min": int,
            "p10": float,
            "median": float,
            "mean": float,
            "p90": float,
            "max": int
        },
        "likes": {...},                 # Same structure
        "comments_count": {...},
        "shares": {...},
        "engagement_total": {...},
        "engagement_rate": {...}
    },
    "correlations": {
        "views_likes": float,           # Pearson r [-1, 1]
        "views_comments": float,
        "views_shares": float,
        "likes_comments": float,
        "likes_shares": float,
        "comments_shares": float,
        "caption_length_engagement": float,
        "hashtag_count_engagement": float
    },
    "top_hashtags": [
        {
            "hashtag": str,
            "total_engagement": int
        },
        ...
    ],
    "top_keywords": [
        {
            "keyword": str,
            "frequency": int
        },
        ...
    ]
}
```

**Engagement Calculations:**
- `engagement_total = likes + comments_count + shares`
- `engagement_rate = engagement_total / views`

**Statistical Methods:**
- Descriptive: mean, median, min, max, p10, p90
- Correlation: Pearson correlation coefficient
- Rankings: sorted by total engagement

---

## 4. CLI Scripts Module

**Location:** `scripts/`  
**Owner:** MLOps Team  
**Purpose:** Command-line interfaces for common operations

### 4.1 validate_data.py

**CLI Signature:**
```bash
python scripts/validate_data.py <jsonl_file_path>
```

**Input:**
- JSONL file path (positional argument)

**Output (stdout):**
```
Validated {n} records; failures: {m}
[Error messages if any]
All records valid. / {m} records failed validation.
```

**Exit Codes:**
- `0`: All records valid
- `1`: Validation errors found

---

### 4.2 build_index.py

**CLI Signature:**
```bash
python scripts/build_index.py
```

**Expected Paths:**
- Input: `data/mock/tiktok_posts_mock.jsonl`
- Output: `data/mock/retrieval_index.pkl`

**Output (stdout):**
```
Loading posts from {path}...
Loaded {n} posts
Building TF-IDF index...
Built index with {n} posts and {m} features
Saving index to {path}...
✓ Index built successfully!
```

**Exit Codes:**
- `0`: Success
- `1`: Error (file not found, validation error, etc.)

---

### 4.3 query_index.py

**CLI Signature:**
```bash
python scripts/query_index.py <query_text> [--topk N] [--json]
```

**Arguments:**
- `query_text`: Space-separated query terms (positional, multi-word)
- `--topk`: Number of results (default: 10)
- `--json`: Output as JSON instead of formatted text

**Input:**
- Pre-built index at `data/mock/retrieval_index.pkl`

**Output (stdout - text mode):**
```
Loading index from {path}...
Query: '{query}'
Top {k} results:

1. {video_id} (score: {score:.4f})
   Caption: {caption}
   Hashtags: {hashtags}
   URL: {url}
...
```

**Output (stdout - JSON mode):**
```json
[
    {
        "video_id": "v001",
        "video_url": "https://...",
        "score": 0.8501,
        "caption": "...",
        "hashtags": ["#tag1", "#tag2"]
    }
]
```

**Exit Codes:**
- `0`: Success
- `1`: Index not found or query error

---

### 4.4 run_baseline.py

**CLI Signature:**
```bash
python scripts/run_baseline.py <jsonl_file_path>
```

**Input:**
- JSONL file path (positional argument)

**Output:**
- Generates `src/baseline/report.md` with formatted markdown report

**stdout:**
```
Wrote baseline report to {path}
```

**Exit Codes:**
- `0`: Success
- `1`: Error reading data or generating report

---

## 5. Module Dependencies

### Dependency Graph

```
┌─────────────────┐
│  Data Layer     │
│  (schemas.py)   │
└────────┬────────┘
         │
         ├──────────────┬──────────────┬─────────────┐
         │              │              │             │
         ▼              ▼              ▼             ▼
┌────────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐
│ Validation │  │  Retrieval   │  │Baseline │  │ Scripts  │
│   Layer    │  │    Module    │  │Analytics│  │  (CLIs)  │
└────────────┘  └──────────────┘  └─────────┘  └──────────┘
```

**Import Rules:**
- All modules depend on `src/common/schemas.py`
- Retrieval module uses schemas for type hints
- Baseline analytics reads TikTokPost objects
- Scripts orchestrate all modules
- No circular dependencies allowed

---

## 6. Future Extensions (Sprint 2+)

### Planned Additions

**6.1 BM25 Retrieval**
- New class: `BM25Index` in `src/retrieval/bm25.py`
- Same interface as `RetrievalIndex`
- Parameters: `k1=1.5`, `b=0.75`

**6.2 Semantic Search (SBERT)**
- New class: `SemanticIndex` in `src/retrieval/semantic.py`
- Input: model_name (e.g., "all-MiniLM-L6-v2")
- Output: Dense vectors instead of sparse TF-IDF

**6.3 Hybrid Retrieval**
- Function: `hybrid_search(bm25_index, semantic_index, query, alpha=0.5)`
- Combines lexical + semantic signals

**6.4 Evaluation Metrics**
- Module: `src/evaluation/metrics.py`
- Functions: `precision_at_k()`, `recall_at_k()`, `ndcg_at_k()`
- Input: ground truth labels + predictions

---

## 7. Contract Versioning

**Current Version:** 1.0 (Sprint 1)

**Change Policy:**
- Breaking changes require major version bump
- New optional parameters are minor version bumps
- Documentation updates are patch version bumps

**Backward Compatibility:**
- All Sprint 1 contracts must remain stable through Sprint 2
- Deprecated functions must have 1-sprint warning period

---

## 8. Testing Requirements

### Contract Validation Tests

Each module must have tests verifying:

1. **Input validation:** Reject invalid inputs gracefully
2. **Output format:** Return correct structure and types
3. **Edge cases:** Empty inputs, single record, large datasets
4. **Error handling:** Proper exceptions with clear messages

### Integration Tests

- End-to-end: JSONL → validation → index → search → results
- Cross-module: Ensure data flows correctly between modules

---

## 9. Performance Contracts

### Latency SLAs (Sprint 1 - Development)

| Operation | Target Latency | Current Performance |
|-----------|----------------|---------------------|
| Validate single record | < 1ms | ~0.5ms |
| Build index (50 posts) | < 5s | ~2s |
| Search query (topk=10) | < 100ms | ~2ms |
| Baseline analytics (50 posts) | < 10s | ~3s |

**Note:** These are development targets. Production SLAs will be defined in Sprint 3.

### Throughput Requirements

- Index building: Support up to 10,000 posts
- Query serving: 10 queries/second minimum
- Validation: 100 records/second minimum

---

## 10. Error Handling Standards

### Exception Hierarchy

```python
# All custom exceptions inherit from base
class TikTokSystemError(Exception):
    """Base exception for all system errors"""
    pass

class ValidationError(TikTokSystemError):
    """Data validation failures"""
    pass

class IndexError(TikTokSystemError):
    """Index building/loading failures"""
    pass

class SearchError(TikTokSystemError):
    """Query execution failures"""
    pass
```

### Error Response Format

All errors should include:
- Clear message describing what went wrong
- Context (e.g., which field failed validation)
- Actionable suggestion for fix (when possible)

---

## Contact and Ownership

| Module | Owner | Email | Slack Channel |
|--------|-------|-------|---------------|
| Data Layer | Ariel | arielo_moreira@hotmail.com | #data-eng |
| Retrieval | Jad | jadchebly@github | #ml-retrieval |
| Baseline | Alp | alparslan@local | #data-science |
| MLOps | Fares | farroseh2005@gmail.com | #mlops |
| Research | Omar | omekkawi.ieu2023@student.ie.edu | #research |

---

**Document Status:** ✅ Active  
**Next Review:** Sprint 2 Planning  
**Approval:** Required for any breaking changes
