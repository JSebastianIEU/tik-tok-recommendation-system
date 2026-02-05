from pathlib import Path

def test_repo_skeleton_integrity():
    expected = [
        "data/mock/tiktok_posts_mock.jsonl",
        "src/common/schemas.py",
        "scripts/validate_data.py",
    ]
    for rel in expected:
        assert Path(rel).exists(), f"Missing {rel}"

# TODO: extend with schema validation and retrieval round-trip tests.
