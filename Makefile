.PHONY: lint test validate-data baseline query datamart train-recommender eval-recommender serve-recommender bootstrap-embedding benchmark-recommender

# Lint only tests for Sprint 1 (repo has known scaffold lint debt elsewhere).
lint:
	python3 -m ruff check tests

# Run only repo tests and ensure src/ is importable.
test:
	PYTHONPATH=. python3 -m pytest -q tests

validate-data:
	PYTHONPATH=. python3 scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl

baseline:
	PYTHONPATH=. python3 scripts/run_baseline.py data/mock/tiktok_posts_mock.jsonl

query:
	PYTHONPATH=. python3 scripts/query_index.py --query "example creator economy" --topk 3

datamart:
	PYTHONPATH=. python3 scripts/build_training_datamart.py data/mock/tiktok_posts_mock.jsonl --output-json data/mock/training_datamart.json

train-recommender:
	PYTHONPATH=. python3 scripts/train_recommender.py data/mock/training_datamart.json --artifact-root artifacts/recommender

eval-recommender:
	PYTHONPATH=. python3 scripts/eval_recommender.py artifacts/recommender/latest --show-manifest

serve-recommender:
	PYTHONPATH=. python3 scripts/serve_recommender.py --host 127.0.0.1 --port 8081 --bundle-dir artifacts/recommender/latest

bootstrap-embedding:
	PYTHONPATH=. python3 scripts/bootstrap_embedding_model.py --model-name sentence-transformers/all-MiniLM-L6-v2

benchmark-recommender:
	PYTHONPATH=. python3 scripts/benchmark_recommender.py --bundle-dir artifacts/recommender/latest --datamart-json data/mock/training_datamart.json --objective engagement --runs 50 --top-k 20 --retrieve-k 200
