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
    parent_comment_id TEXT REFERENCES comments(comment_id)
);

CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);

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

