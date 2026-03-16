## Notebooks

Conventions:

- Name notebooks by objective and sequence, e.g. `01_data_quality.ipynb`, `02_comments_patterns.ipynb`.
- Treat notebooks as analysis layers over `../extracts/bronze/<run_id>/` or `../extracts/silver/`.
- Do not embed credentials or direct DB passwords in notebooks.
- Export stable plots/tables to `../reports/`.
- Prefer headless execution via `eda/src/notebook_runner.py` (`papermill`) when automating.

Recommended notebook header cells:

1. `RUN_ID` and source manifest path.
2. Data loading paths.
3. Assumptions and known caveats.
