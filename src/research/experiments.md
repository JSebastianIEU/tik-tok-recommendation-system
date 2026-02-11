# Retrieval Method Experiments

## Experiment 1: BM25 vs TF-IDF on Mock TikTok Data

### Objective
Compare retrieval quality and speed of keyword-based methods on short-form social content.

### Dataset
Used 5 mock TikTok videos from the project's mock data (`data/mock/tiktok_posts_mock.jsonl`).

Videos covered diverse content categories:
- Fitness (core workout)
- Food (ramen recipe hack)
- Travel (Kyoto autumn guide)
- Tech (React tutorial)
- Sustainability (composting)

Each video contained:
- Caption
- Hashtags
- Keywords
- Metadata (likes, views, comments)

### Indexing Strategy
Combined caption, hashtags, and keywords into a single searchable text field. This approach ensures all relevant metadata is considered during retrieval.

### Test Queries

1. "core workout exercises" - Should return fitness content
2. "easy meal recipes" - Should return food/cooking content
3. "japan travel tips" - Should return travel content
4. "coding tutorial" - Should return tech/programming content

### Methodology

Used Python libraries:
- `rank-bm25` for BM25 implementation
- `scikit-learn` for TF-IDF vectorization

Both methods searched the combined text (caption + hashtags + keywords) and returned top 3 results with relevance scores.

### Results

#### Query 1: "core workout exercises"

**BM25 Results (0.00ms):**
1. [v001] 5-minute core burner (score: 2.819)
2. [v002] 3-ingredient ramen hack (score: 0.000)
3. [v003] Kyoto hidden gems in autumn (score: 0.000)

**TF-IDF Results (2.00ms):**
1. [v001] 5-minute core burner (score: 0.559)
2. [v005] Turn kitchen scraps into compost (score: 0.000)
3. [v004] Build a to-do app with React in 60s (score: 0.000)

**Observation:** Both methods correctly identified the fitness video as most relevant. BM25 showed stronger score differentiation (2.819 vs 0.000) compared to TF-IDF (0.559 vs 0.000).

#### Query 2: "easy meal recipes"

**BM25 Results (0.00ms):**
All results scored 0.000 - no keyword matches found.

**TF-IDF Results (1.57ms):**
All results scored 0.000 - no keyword matches found.

**Observation:** Neither method found relevant results. The ramen video (v002) should have matched but uses keywords like "ramen hack" and "budget meals" rather than "recipe" or "meal". This demonstrates a limitation of keyword-based retrieval - it fails when users phrase queries differently than content creators tag their videos.

#### Query 3: "japan travel tips"

**BM25 Results (0.00ms):**
1. [v003] Kyoto hidden gems in autumn (score: 3.296)
2. [v001] 5-minute core burner (score: 0.000)
3. [v002] 3-ingredient ramen hack (score: 0.000)

**TF-IDF Results (1.96ms):**
1. [v003] Kyoto hidden gems in autumn (score: 0.406)
2. [v005] Turn kitchen scraps into compost (score: 0.000)
3. [v004] Build a to-do app with React in 60s (score: 0.000)

**Observation:** Both methods correctly identified the travel video. BM25 again showed stronger score separation. The match succeeded because the video's keywords field contained "Japan tips" and hashtags included "travel".

#### Query 4: "coding tutorial"

**BM25 Results (0.00ms):**
1. [v004] Build a to-do app with React in 60s (score: 1.002)
2. [v001] 5-minute core burner (score: 0.000)
3. [v002] 3-ingredient ramen hack (score: 0.000)

**TF-IDF Results (1.00ms):**
1. [v004] Build a to-do app with React in 60s (score: 0.237)
2. [v005] Turn kitchen scraps into compost (score: 0.000)
3. [v003] Kyoto hidden gems in autumn (score: 0.000)

**Observation:** Both methods correctly identified the React tutorial video. The match worked due to keywords like "react tutorial" and hashtags "#coding".

### Speed Comparison

| Query | BM25 Time | TF-IDF Time | Winner |
|-------|-----------|-------------|--------|
| "core workout exercises" | 0.00ms | 2.00ms | BM25 |
| "easy meal recipes" | 0.00ms | 1.57ms | BM25 |
| "japan travel tips" | 0.00ms | 1.96ms | BM25 |
| "coding tutorial" | 0.00ms | 1.00ms | BM25 |
| **Average** | **0.00ms** | **1.63ms** | **BM25** |

BM25 consistently outperformed TF-IDF in speed, though both were fast enough for real-time queries on this small dataset.

### Key Findings

#### BM25 Strengths
1. Faster than TF-IDF (effectively instant on small datasets)
2. Better score differentiation between relevant and irrelevant results
3. Successfully matched 3 out of 4 queries when relevant keywords existed
4. Clear interpretation - can see which keywords drove the match

#### Limitations of Keyword-Based Retrieval (Both Methods)
1. **Vocabulary mismatch:** Query "easy meal recipes" failed to match "ramen hack" even though semantically relevant
2. **Dependency on metadata quality:** Success relies on creators using searchable hashtags and keywords
3. **No semantic understanding:** Cannot infer that "ramen hack" is a type of "meal recipe"
4. **Exact matching only:** Synonyms, paraphrasing, and related concepts are missed

#### When Keyword Methods Work Well
- User queries match content creator vocabulary (e.g., "coding tutorial" matches "#coding")
- Content has rich, well-tagged metadata
- Queries use common, specific terms

#### When Keyword Methods Fail
- Vocabulary mismatch between users and creators
- Abstract or conceptual queries
- Queries using synonyms or paraphrasing
- Slang, abbreviations, or colloquialisms

### Precision Analysis

| Query | Relevant Video Exists? | BM25 Found It? | TF-IDF Found It? |
|-------|------------------------|----------------|------------------|
| "core workout exercises" | Yes (v001) | Yes | Yes |
| "easy meal recipes" | Yes (v002) | No | No |
| "japan travel tips" | Yes (v003) | Yes | Yes |
| "coding tutorial" | Yes (v004) | Yes | Yes |
| **Success Rate** | - | **75%** | **75%** |

Both methods achieved 75% precision on this test set, with identical failures due to vocabulary mismatch.

### Recommendations

#### For Sprint 2: Implement BM25 Baseline
1. Use BM25 as the initial retrieval method
2. Index combined text (caption + hashtags + keywords)
3. Use default parameters (k1=1.5, b=0.75)
4. Accept that approximately 25% of queries may fail due to vocabulary mismatch

#### For Sprint 3: Address Semantic Gaps
The "easy meal recipes" failure demonstrates the need for semantic understanding. Options:
1. **Synonym expansion:** Manually map "recipes" to ["recipe", "hack", "meal", "cooking"]
2. **Query rewriting:** Use LLM to expand user queries before retrieval
3. **Semantic embeddings:** Add Sentence-BERT for queries that fail BM25
4. **Hybrid approach:** Use BM25 as first-pass filter, then re-rank with embeddings

#### Success Metrics for Production
- Precision at 5 greater than 80% (currently 75% on limited test)
- Query latency under 10ms for 95th percentile
- Fallback mechanism for zero-result queries (suggest related content or popular videos)

### Limitations of This Experiment

1. **Small dataset:** Only 5 videos tested - production would have thousands or millions
2. **Limited query diversity:** Only 4 test queries - real users generate hundreds of query patterns
3. **No user feedback:** Cannot measure actual engagement or satisfaction
4. **Manual evaluation:** No automated metrics or A/B testing framework

### Next Steps

1. Expand test dataset to 50-100 videos across more categories
2. Collect real user queries from analytics or user research
3. Implement BM25 in production with monitoring
4. Track queries that return zero or low-quality results
5. Use failed queries to inform Sprint 3 semantic retrieval implementation

## Conclusion

BM25 is a suitable baseline for Sprint 2 with 75% precision on keyword-matched queries and sub-millisecond latency. However, the experiment clearly demonstrates the need for semantic search capabilities in future sprints to handle vocabulary mismatch cases.