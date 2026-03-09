## Team Runbook (JSON-Only)

No PostgreSQL or Docker is required for this runbook.

## 1) Setup (all members)

PowerShell:

```powershell
pip install -r scraper/requirements.txt
python -m playwright install
```

macOS:

```bash
pip install -r scraper/requirements.txt
python -m playwright install
```

Optional:

- set `MS_TOKEN` for better TikTokApi discovery.

## 2) Run each member config

```bash
python -m scraper run --config "scraper/configs/juan_sebastian.yaml"
python -m scraper run --config "scraper/configs/jad_chebly.yaml"
python -m scraper run --config "scraper/configs/omar_mekawi.yaml"
python -m scraper run --config "scraper/configs/alp_arslan.yaml"
python -m scraper run --config "scraper/configs/arielo_moreira.yaml"
python -m scraper run --config "scraper/configs/fares_rafiq.yaml"
```

Each run writes a separate JSONL file in `scraper/data/raw/`.

## 3) Merge all JSONL files

```bash
python -m scraper merge-json \
  --output "scraper/data/raw/team_merged.jsonl" \
  --input "scraper/data/raw/juan_sebastian_fitness_self_improvement.jsonl" \
  --input "scraper/data/raw/jad_chebly_finance_investing.jsonl" \
  --input "scraper/data/raw/omar_mekawi_relationships_dating.jsonl" \
  --input "scraper/data/raw/alp_arslan_politics_social_issues.jsonl" \
  --input "scraper/data/raw/arielo_moreira_lifestyle_luxury_aspirational.jsonl" \
  --input "scraper/data/raw/fares_rafiq_entertainment_trends_viral.jsonl"
```

