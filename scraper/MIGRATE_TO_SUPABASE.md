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

## Step 2: Get Your Supabase Connection String

1. Open [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Project Settings** (gear icon) → **Database**
4. Under **Connection string**, choose **URI**
5. Copy the URI. It looks like:
   ```
   postgresql://postgres.[PROJECT_REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
   ```
   Or for direct connection:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
   ```
6. Replace `[YOUR-PASSWORD]` with your actual database password (from the same page)

---

## Step 3: Import Data into Supabase

```bash
/opt/homebrew/opt/libpq/bin/psql "YOUR_SUPABASE_URI" < backup.sql
```

Example (replace `YOUR_PASSWORD` and project ref if different):

```bash
/opt/homebrew/opt/libpq/bin/psql "postgresql://postgres:YOUR_PASSWORD@db.mlmlcilyoqvbvgljsjtv.supabase.co:5432/postgres" < backup.sql
```

- If you see errors about tables already existing, that’s usually fine (schema was already applied)
- Data should now be in Supabase

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
   python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing
   ```

New scrapes will go to Supabase. `--skip-existing` skips videos already in the DB (including migrated ones).

---

## Summary

| Step | Action |
|------|--------|
| 1 | `docker exec tiktok-postgres pg_dump -U tiktok tiktok > backup.sql` |
| 2 | Get Supabase URI from Dashboard → Settings → Database |
| 3 | `psql "SUPABASE_URI" < backup.sql` |
| 4 | Set `DATABASE_URL` in `scraper/.env` |
| 5 | `python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing` |

---

## Troubleshooting

- **`psql: command not found`** – Add to PATH: `export PATH="/opt/homebrew/opt/libpq/bin:$PATH"`
- **Connection refused** – Check Supabase project is running and URI is correct
- **Password special characters** – URL-encode them in the connection string (e.g. `@` → `%40`)
- **Schema conflicts** – If restore fails on existing objects, you can init Supabase first with `python -m scraper init-db` (using Supabase `DATABASE_URL`), then restore only data, or ignore schema errors if data was restored
