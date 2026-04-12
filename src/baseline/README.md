# Baseline Module

> Status note: this directory contains lightweight baseline analytics and report generation for mock/local workflows.
> It is not the main entrypoint for the current recommender platform.

- `baseline_stats.py`: descriptive statistics and report generation over mock JSONL data (engagement metrics match `src/common/schemas.py`)
- `report.md`: generated markdown report output

Use this module for simple exploratory checks and baseline summaries.
For the main training and serving path, use `src/recommendation/`.
