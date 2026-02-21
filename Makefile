.PHONY: lint test validate-data baseline query

# Lint only tests for Sprint 1 (repo has known scaffold lint debt elsewhere).
lint:
	python -m ruff check tests

# Run only repo tests and ensure src/ is importable.
test:
	PYTHONPATH=. python -m pytest -q tests

validate-data:
	PYTHONPATH=. python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl

baseline:
	PYTHONPATH=. python scripts/run_baseline.py data/mock/tiktok_posts_mock.jsonl

query:
	PYTHONPATH=. python scripts/query_index.py --query "example creator economy" --topk 3
