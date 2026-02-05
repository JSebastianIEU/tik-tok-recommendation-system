.PHONY: lint test validate-data baseline query

lint:
	@echo "TODO: add linter (e.g., ruff/flake8) and config, then invoke it here"

test:
	pytest

validate-data:
	python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl

baseline:
	python scripts/run_baseline.py data/mock/tiktok_posts_mock.jsonl

query:
	python scripts/query_index.py --query "example creator economy" --topk 3
