## Notebooks

### Main EDA Notebook

**`eda_tiktok.py`** — Comprehensive Marimo notebook covering all EDA for the TikTok dataset.

Run interactively:
```bash
DATABASE_URL="postgresql://..." marimo edit eda/notebooks/eda_tiktok.py
```

Export to static HTML (for the report):
```bash
DATABASE_URL="postgresql://..." marimo export html eda/notebooks/eda_tiktok.py -o eda/reports/latest/eda_tiktok.html --no-include-code
```

Sections covered:
1. Database connection and raw extraction (31K+ videos, 22K+ authors, 19K+ comments)
2. Data quality and completeness (null rates, zero-engagement, duplicates)
3. Temporal distribution (monthly trends, day-of-week patterns)
4. Engagement metrics analysis (distributions, correlations, engagement rates)
5. Content analysis (duration, caption length, duration vs engagement)
6. Creator landscape (power-law dynamics, follower distribution)
7. Hashtag and topic analysis (Zipf's law, coverage)
8. Comment analysis (volume, text length, threading)
9. Verified vs non-verified creator performance
10. Key findings and implications for the recommendation system

### Conventions

- Do not embed credentials or direct DB passwords in notebooks. Use `DATABASE_URL` env var.
- Export stable plots/tables to `../reports/`.
- Marimo notebooks are Python files (`.py`) with reactive cell dependencies.
