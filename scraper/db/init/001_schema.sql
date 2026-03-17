-- TikTok 3NF schema for scraped data

CREATE TABLE IF NOT EXISTS authors (
    author_id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    display_name TEXT,
    bio TEXT,
    avatar_url TEXT,
    verified BOOLEAN
);

CREATE TABLE IF NOT EXISTS audios (
    audio_id TEXT PRIMARY KEY,
    audio_name TEXT NOT NULL,
    audio_author_name TEXT,
    is_original BOOLEAN
);

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL REFERENCES authors(author_id),
    url TEXT UNIQUE NOT NULL,
    caption TEXT,
    duration_sec INTEGER,
    thumbnail_url TEXT,
    created_at TIMESTAMPTZ,
    audio_id TEXT REFERENCES audios(audio_id)
);

CREATE TABLE IF NOT EXISTS hashtags (
    hashtag_id SERIAL PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS video_hashtags (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    hashtag_id INTEGER NOT NULL REFERENCES hashtags(hashtag_id) ON DELETE CASCADE,
    PRIMARY KEY (video_id, hashtag_id)
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    scrape_run_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_snapshots (
    video_snapshot_id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    scraped_at TIMESTAMPTZ NOT NULL,
    likes BIGINT,
    comments_count BIGINT,
    shares BIGINT,
    plays BIGINT,
    scrape_run_id UUID REFERENCES scrape_runs(scrape_run_id),
    position INTEGER,
    UNIQUE (video_id, scraped_at)
);

CREATE INDEX IF NOT EXISTS idx_video_snapshots_video_id ON video_snapshots(video_id);
CREATE INDEX IF NOT EXISTS idx_video_snapshots_scraped_at ON video_snapshots(scraped_at);
CREATE INDEX IF NOT EXISTS idx_video_snapshots_scrape_run_id ON video_snapshots(scrape_run_id);

CREATE TABLE IF NOT EXISTS author_metric_snapshots (
    author_metric_snapshot_id BIGSERIAL PRIMARY KEY,
    author_id TEXT NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    video_id TEXT REFERENCES videos(video_id) ON DELETE SET NULL,
    scraped_at TIMESTAMPTZ NOT NULL,
    follower_count BIGINT,
    following_count BIGINT,
    author_likes_count BIGINT,
    UNIQUE (author_id, video_id, scraped_at)
);

CREATE INDEX IF NOT EXISTS idx_author_metric_snapshots_author_id ON author_metric_snapshots(author_id);

CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    author_id TEXT REFERENCES authors(author_id),
    username TEXT,
    text TEXT NOT NULL,
    parent_comment_id TEXT REFERENCES comments(comment_id),
    root_comment_id TEXT,
    comment_level SMALLINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);

ALTER TABLE comments ADD COLUMN IF NOT EXISTS root_comment_id TEXT;
ALTER TABLE comments ADD COLUMN IF NOT EXISTS comment_level SMALLINT NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_comments_root_comment_id ON comments(root_comment_id);
CREATE INDEX IF NOT EXISTS idx_comments_comment_level ON comments(comment_level);

WITH RECURSIVE comment_lineage AS (
    SELECT
        c.comment_id,
        c.parent_comment_id,
        c.comment_id AS root_comment_id,
        0::SMALLINT AS comment_level
    FROM comments c
    WHERE c.parent_comment_id IS NULL
    UNION ALL
    SELECT
        child.comment_id,
        child.parent_comment_id,
        lineage.root_comment_id,
        (lineage.comment_level + 1)::SMALLINT AS comment_level
    FROM comments child
    JOIN comment_lineage lineage ON child.parent_comment_id = lineage.comment_id
    WHERE lineage.comment_level < 32
),
resolved_lineage AS (
    SELECT
        c.comment_id,
        COALESCE(
            l.root_comment_id,
            CASE
                WHEN c.parent_comment_id IS NULL THEN c.comment_id
                ELSE c.parent_comment_id
            END
        ) AS root_comment_id,
        COALESCE(
            l.comment_level,
            CASE
                WHEN c.parent_comment_id IS NULL THEN 0
                ELSE 1
            END
        )::SMALLINT AS comment_level
    FROM comments c
    LEFT JOIN comment_lineage l ON l.comment_id = c.comment_id
)
UPDATE comments c
SET
    root_comment_id = r.root_comment_id,
    comment_level = r.comment_level
FROM resolved_lineage r
WHERE c.comment_id = r.comment_id
  AND (
      c.root_comment_id IS DISTINCT FROM r.root_comment_id
      OR c.comment_level IS DISTINCT FROM r.comment_level
  );

CREATE TABLE IF NOT EXISTS comment_snapshots (
    comment_snapshot_id BIGSERIAL PRIMARY KEY,
    comment_id TEXT NOT NULL REFERENCES comments(comment_id) ON DELETE CASCADE,
    video_snapshot_id BIGINT NOT NULL REFERENCES video_snapshots(video_snapshot_id) ON DELETE CASCADE,
    scraped_at TIMESTAMPTZ NOT NULL,
    likes BIGINT,
    reply_count BIGINT,
    UNIQUE (comment_id, video_snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_comment_snapshots_comment_id ON comment_snapshots(comment_id);
CREATE INDEX IF NOT EXISTS idx_comment_snapshots_video_snapshot_id ON comment_snapshots(video_snapshot_id);

CREATE TABLE IF NOT EXISTS scrape_source_jobs (
    source_key TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scrape_source_jobs_status ON scrape_source_jobs(status);

CREATE TABLE IF NOT EXISTS comment_enrichment_jobs (
    video_id TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_attempt_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comment_enrichment_jobs_status ON comment_enrichment_jobs(status);
CREATE INDEX IF NOT EXISTS idx_comment_enrichment_jobs_next_retry ON comment_enrichment_jobs(next_retry_at);
