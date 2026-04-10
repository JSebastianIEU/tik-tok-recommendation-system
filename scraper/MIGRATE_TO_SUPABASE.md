# Migrate Local Postgres → Supabase

Step-by-step guide to move your scraped data to Supabase and point the scraper there.

---

## What You Need Before Starting

1. **Supabase project** – Create one at [supabase.com](https://supabase.com) if you don’t have it
2. **Supabase database password** – From Supabase → Project Settings → Database → Database password
3. **Local Docker Postgres running** – `cd scraper && docker compose up -d`
4. **`psql` installed** – `brew install libpq` (then `export PATH="/opt/homebrew/opt/libpq/bin:$PATH"`)

---

## Step 1: Export Data from Local Postgres

From your project root:

```bash
docker exec tiktok-postgres pg_dump -U tiktok tiktok > backup.sql
```

- **Source**: Local Docker Postgres
- **Output**: `backup.sql` in the current directory
- **Effect**: Read-only; local data is unchanged

---

## Step 2: Get Your Supabase Connection String (psql)

1. Open [Supabase Dashboard](https://supabase.com/dashboard) → your project
2. Go to **Project Settings** (gear icon) → **Database**
3. Scroll to **Connection string**
4. Select **URI**
5. **Use the pooler** (recommended if direct connection fails with "No route to host"):
   - Toggle **Use connection pooling** ON
   - Mode: **Transaction**
   - Copy the URI. Format:
     ```
     postgresql://postgres.[PROJECT_REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
     ```
6. Or use **direct** (if your network allows it):
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
   ```
7. Replace `[YOUR-PASSWORD]` with your database password (same page, under "Database password")

---

## Step 3: Import Data with psql

From your project root (where `backup.sql` is):

```bash
psql "YOUR_FULL_URI" < backup.sql
```

If `psql` isn't in PATH:

```bash
/opt/homebrew/opt/libpq/bin/psql "YOUR_FULL_URI" < backup.sql
```

**Example (pooler):**
```bash
psql "postgresql://postgres.mlmlcilyoqvbvgljsjtv:YOUR_PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres" < backup.sql
```

**Example (direct):**
```bash
psql "postgresql://postgres:YOUR_PASSWORD@db.mlmlcilyoqvbvgljsjtv.supabase.co:5432/postgres" < backup.sql
```

- Replace `YOUR_PASSWORD` with your actual password
- Replace `us-east-1` with your region if different (check the URI from the dashboard)
- Some "relation already exists" errors are OK if schema was applied before

---

## Step 4: Point the Scraper to Supabase

Edit `scraper/.env`:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@db.mlmlcilyoqvbvgljsjtv.supabase.co:5432/postgres
MS_TOKEN=your_ms_token_here
```

Use the same Supabase connection string you used in Step 3.

---

## Step 5: Verify and Run Scraper

1. **Check connection**:
   ```bash
   python -m scraper init-db
   ```
   If the schema already exists, this may report that tables exist. That’s fine.

2. **Scrape into Supabase**:
   ```bash
   python -m scraper scrape-all scraper/configs/full_scale.yaml
   ```

New scrapes will go to Supabase. Default behavior updates/upserts existing videos so snapshot data stays fresh. Add `--skip-existing` only for insert-only runs.

---

## Summary

| Step | Action |
|------|--------|
| 1 | `docker exec tiktok-postgres pg_dump -U tiktok tiktok > backup.sql` |
| 2 | Get Supabase URI from Dashboard → Settings → Database |
| 3 | `psql "SUPABASE_URI" < backup.sql` |
| 4 | Set `DATABASE_URL` in `scraper/.env` |
| 5 | `python -m scraper scrape-all scraper/configs/full_scale.yaml` |

---

## Troubleshooting

- **`psql: command not found`** – Add to PATH: `export PATH="/opt/homebrew/opt/libpq/bin:$PATH"`
- **Connection refused** – Check Supabase project is running and URI is correct
- **Password special characters** – URL-encode them in the connection string (e.g. `@` → `%40`)
- **Schema conflicts** – If restore fails on existing objects, you can init Supabase first with `python -m scraper init-db` (using Supabase `DATABASE_URL`), then restore only data, or ignore schema errors if data was restored
