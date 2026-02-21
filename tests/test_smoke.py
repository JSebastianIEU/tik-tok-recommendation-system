from pathlib import Path

from src.common.validation import load_jsonl
from src.retrieval.index import RetrievalIndex


def test_repo_skeleton_integrity():
    expected = [
        "data/mock/tiktok_posts_mock.jsonl",
        "src/common/schemas.py",
        "scripts/validate_data.py",
    ]
    for rel in expected:
        assert Path(rel).exists(), f"Missing {rel}"


def test_imports_and_basic_instantiation():
    """
    Smoke test: verify core modules import and basic objects can be created.
    """
    index = RetrievalIndex()
    assert index is not None


def test_mock_data_has_50_records():
    """
    Smoke test: ensure mocked dataset is stable and has exactly 50 records.
    """
    data_path = Path("data/mock/tiktok_posts_mock.jsonl")
    records = list(load_jsonl(data_path))
    assert len(records) == 50

