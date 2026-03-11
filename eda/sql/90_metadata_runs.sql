-- Optional table template if you later store EDA run metadata in Postgres.
CREATE TABLE IF NOT EXISTS eda_run_metadata (
    run_id TEXT PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    plan_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    git_sha TEXT,
    db_host TEXT,
    payload JSONB NOT NULL
);
