
from pathlib import Path

from src.common.validation import load_jsonl, validate_record, validate_stream


def test_load_jsonl_counts():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    records = list(load_jsonl(str(path)))
    assert len(records) >= 20


def test_validate_sample_record():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    rec = next(load_jsonl(str(path)))
    ok, errors = validate_record(rec)
    assert ok, f"Sample record failed validation: {errors}"


def test_validate_stream_all():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    records = list(load_jsonl(str(path)))
    total, failed, failures = validate_stream(records)
    assert total == len(records)
    assert failed == 0, f"Some records failed validation: {failures}"
