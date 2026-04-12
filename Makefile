.PHONY: lint test validate-data generate-mock baseline datamart feature-snapshot fabric-hourly fabric-daily comment-snapshot comment-priors comment-hourly comment-daily eval-fabric-rollout eval-comment-rollout train-recommender eval-recommender eval-recommender-ablation serve-recommender refresh-serving-bundle bootstrap-embedding benchmark-recommender build-retriever fit-retriever-fusion eval-retriever-only eval outcome-attribution drift-monitor retrain-controller experiment-analysis live-e2e-validate

# Lint only tests for Sprint 1 (repo has known scaffold lint debt elsewhere).
lint:
	python3 -m ruff check tests

# Run only repo tests and ensure src/ is importable.
test:
	PYTHONPATH=. python3 -m pytest -q tests

generate-mock:
	PYTHONPATH=. python3 -m src.data.mock_generator --count 50 --seed 42

validate-data:
	PYTHONPATH=. python3 scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl

baseline:
	PYTHONPATH=. python3 scripts/run_baseline.py data/mock/tiktok_posts_mock.jsonl

datamart:
	PYTHONPATH=. python3 scripts/build_training_datamart.py data/mock/tiktok_posts_mock.jsonl --output-json data/mock/training_datamart.json

feature-snapshot:
	PYTHONPATH=. python3 scripts/build_feature_fabric_snapshot.py data/mock/tiktok_posts_mock.jsonl --contract-manifest-root artifacts/contracts --output-root artifacts/features --mode full

fabric-hourly:
	PYTHONPATH=. python3 scripts/run_feature_fabric_cadence.py data/mock/tiktok_posts_mock.jsonl --mode hourly_incremental --contracts-root artifacts/contracts --features-root artifacts/features

fabric-daily:
	PYTHONPATH=. python3 scripts/run_feature_fabric_cadence.py data/mock/tiktok_posts_mock.jsonl --mode daily_full --contracts-root artifacts/contracts --features-root artifacts/features

comment-snapshot:
	PYTHONPATH=. python3 scripts/build_comment_intelligence_snapshot.py data/mock/tiktok_posts_mock.jsonl --contract-manifest-root artifacts/contracts --output-root artifacts/comment_intelligence/features --mode full

comment-priors:
	PYTHONPATH=. python3 scripts/build_comment_transfer_priors.py $(MANIFEST) --output-root artifacts/comment_intelligence/priors

comment-hourly:
	PYTHONPATH=. python3 scripts/run_comment_intelligence_cadence.py data/mock/tiktok_posts_mock.jsonl --mode hourly_incremental --contracts-root artifacts/contracts --features-root artifacts/comment_intelligence/features --priors-root artifacts/comment_intelligence/priors

comment-daily:
	PYTHONPATH=. python3 scripts/run_comment_intelligence_cadence.py data/mock/tiktok_posts_mock.jsonl --mode daily_full --contracts-root artifacts/contracts --features-root artifacts/comment_intelligence/features --priors-root artifacts/comment_intelligence/priors

eval-fabric-rollout:
	PYTHONPATH=. python3 scripts/eval_fabric_rollout.py artifacts/features/baseline_metrics.json artifacts/features/shadow_metrics.json

eval-comment-rollout:
	PYTHONPATH=. python3 scripts/eval_comment_intelligence_rollout.py artifacts/comment_intelligence/baseline_metrics.json artifacts/comment_intelligence/shadow_metrics.json

train-recommender:
	PYTHONPATH=. python3 scripts/train_recommender.py data/mock/training_datamart.json --artifact-root artifacts/recommender

eval-recommender:
	PYTHONPATH=. python3 scripts/eval_recommender.py artifacts/recommender/latest --show-manifest

eval-recommender-ablation:
	PYTHONPATH=. python3 scripts/eval_recommender.py artifacts/recommender/latest --show-ablation

serve-recommender:
	PYTHONPATH=. python3 scripts/serve_recommender.py --host 127.0.0.1 --port 8081 --bundle-dir artifacts/recommender_real/latest

refresh-serving-bundle:
	PYTHONPATH=. python3 scripts/refresh_serving_bundle.py --db-url $$DATABASE_URL

bootstrap-embedding:
	PYTHONPATH=. python3 scripts/bootstrap_embedding_model.py --model-name sentence-transformers/all-MiniLM-L6-v2

benchmark-recommender:
	PYTHONPATH=. python3 scripts/benchmark_recommender.py --bundle-dir artifacts/recommender/latest --datamart-json data/mock/training_datamart.json --objective engagement --runs 50 --top-k 20 --retrieve-k 200

build-retriever:
	PYTHONPATH=. python3 scripts/build_retriever_index.py data/mock/training_datamart.json --output-dir artifacts/retriever/latest

fit-retriever-fusion:
	PYTHONPATH=. python3 scripts/fit_retriever_fusion.py data/mock/training_datamart.json --retriever-dir artifacts/retriever/latest

eval-retriever-only:
	PYTHONPATH=. python3 scripts/eval_retriever.py data/mock/training_datamart.json --retriever-dir artifacts/retriever/latest

eval:
	@echo "Usage: make eval MODE=recommender|retriever|benchmark ARGS='...'"
	@echo "Example: make eval MODE=recommender ARGS='artifacts/recommender/latest --show-manifest'"
	PYTHONPATH=. python3 scripts/run_eval.py --mode $(MODE) $(ARGS)

outcome-attribution:
	PYTHONPATH=. python3 scripts/run_outcome_attribution.py --output-json artifacts/control_plane/outcome_attribution_report.json

drift-monitor:
	PYTHONPATH=. python3 scripts/run_drift_monitor.py --output-json artifacts/control_plane/drift_report.json

retrain-controller:
	PYTHONPATH=. python3 scripts/run_retrain_controller.py --output-json artifacts/control_plane/retrain_decision.json

experiment-analysis:
	PYTHONPATH=. python3 scripts/run_experiment_analysis.py --output-json artifacts/control_plane/experiment_report.json

live-e2e-validate:
	PYTHONPATH=. python3 scripts/run_live_e2e_validation.py --launch-python --launch-node --inject-compat-mismatch --inject-python-down --run-control-plane --db-url $$DATABASE_URL --output-json artifacts/control_plane/live_e2e_validation_report.json
