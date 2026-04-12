Data schema for mocked TikTok-like posts
======================================

> Status note: this document describes the older mock JSONL schema used by the baseline scaffold modules in `src/common/`, `src/data/`, and `src/baseline/`.
> It does not describe the full canonical recommender contract in `src/recommendation/contracts.py`.

**Structured validation** lives in `src/common/schemas.py` (`TikTokPost`, etc.): hashtags are normalized to `#token` form, `video_meta.language` must match an ISO-like tag (e.g. `en`, `pt-br`), and each post exposes **`engagement_total`** and **`engagement_rate`** (computed fields) using the same formulas as `scripts/validate_data.py` and the baseline report (`compute_engagement_total` / `compute_engagement_rate`).


Top-level fields (required)
- `video_id`: string — unique id for the video (e.g. "v001").
- `video_url`: string (URL) — link to the video page.
- `caption`: string — short textual caption for the video.
- `hashtags`: array[string] — list of hashtag strings (including `#`).
- `keywords`: array[string] — list of search / topic keywords.
- `search_query`: string — the originating search query that surfaced the video.
- `posted_at`: ISO 8601 timestamp string — when the video was posted (UTC recommended).
- `likes`: integer >= 0 — total likes.
- `comments_count`: integer >= 0 — total comment count (should match `len(comments)` in mocks).
- `shares`: integer >= 0 — total shares.
- `views`: integer >= 0 — total view count.

Nested objects
- `author`: object with
  - `author_id`: string
  - `username`: string
  - `followers`: integer >= 0
- `audio`: object with
  - `audio_id`: string
  - `audio_title`: string
- `video_meta`: object with
  - `duration_seconds`: integer > 0
  - `language`: string (ISO language code or short label)
- `comments`: array of comment objects (minimum 3 in mock data) — each comment object contains:
  - `comment_id`: string
  - `text`: string
  - `likes`: integer >= 0
  - `created_at`: ISO 8601 timestamp string

Constraints and notes
- All numeric counters (`likes`, `shares`, `views`, `followers`, comment `likes`) must be non-negative integers.
- `video_meta.duration_seconds` must be positive.
- `posted_at` and comment `created_at` should be valid ISO 8601 datetimes; UTC-aware timestamps are preferred (e.g. `2025-11-18T14:22:00Z`).
- In mock data, `comments_count` should reflect the number of items in the `comments` array; downstream systems may not require exact equality but it is useful for testing.

Example record (compact):

{
  "video_id": "v001",
  "video_url": "https://www.tiktok.com/@fitjules/video/700000001",
  "caption": "5-minute core burner",
  "hashtags": ["#fitness", "#core"],
  "keywords": ["core workout", "plank"],
  "search_query": "at home core workout",
  "posted_at": "2025-11-18T14:22:00Z",
  "likes": 18450,
  "comments_count": 3,
  "shares": 420,
  "views": 256000,
  "author": {"author_id": "a901", "username": "fitjules", "followers": 482000},
  "audio": {"audio_id": "au101", "audio_title": "Upbeat Gym Mix"},
  "video_meta": {"duration_seconds": 38, "language": "en"},
  "comments": [
    {"comment_id": "c1", "text": "Tried this!", "likes": 120, "created_at": "2025-11-18T15:10:00Z"},
    {"comment_id": "c2", "text": "Nice routine", "likes": 45, "created_at": "2025-11-18T15:35:00Z"},
    {"comment_id": "c3", "text": "Great pace", "likes": 33, "created_at": "2025-11-18T16:05:00Z"}
  ]
}

# Data Module

- `mock_generator.py`: helper for synthetic/mock records used in local experimentation
- data sources for this folder are mock/external inputs, not the main canonical recommender pipeline

For the current canonical contract layer, use `src/recommendation/contracts.py`.
