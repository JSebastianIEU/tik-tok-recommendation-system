from pathlib import Path

from src.common.schemas import (
    TikTokPost,
    compute_engagement_rate,
    compute_engagement_total,
)
from src.common.validation import (
    format_failure_details,
    format_validation_summary,
    load_jsonl,
    validate_file,
    validate_jsonl_file,
    validate_record,
    validate_stream,
)


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


def test_validate_jsonl_file_matches_stream_for_valid_fixture():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    records = list(load_jsonl(str(path)))
    t1, f1, fail1 = validate_stream(records)
    t2, f2, fail2 = validate_jsonl_file(path)
    assert t1 == t2
    assert f1 == f2
    assert fail1 == fail2


def test_validate_jsonl_file_reports_json_syntax_error(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    total, failed, failures = validate_jsonl_file(bad)
    assert total == 1
    assert failed == 1
    assert failures[0][0] == 1
    assert any("json:" in e for e in failures[0][1])


def test_validate_file_flat_errors(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    passed, errors = validate_file(bad)
    assert passed == 0
    assert len(errors) == 1
    assert errors[0].startswith("Line 1:")


def test_engagement_helpers_match_tiktok_post_computed_fields():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    rec = next(load_jsonl(str(path)))
    post = TikTokPost.model_validate(rec)
    assert post.engagement_total == compute_engagement_total(
        likes=post.likes,
        comments_count=post.comments_count,
        shares=post.shares,
    )
    assert post.engagement_rate == compute_engagement_rate(
        likes=post.likes,
        comments_count=post.comments_count,
        shares=post.shares,
        views=post.views,
    )


def test_schema_rejects_bad_hashtag():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    rec = dict(next(load_jsonl(str(path))))
    rec["hashtags"] = ["bad tag"]
    err = validate_record(rec)
    assert not err[0]


def test_schema_normalizes_language_case():
    path = Path("data/mock/tiktok_posts_mock.jsonl")
    rec = dict(next(load_jsonl(str(path))))
    rec["video_meta"] = dict(rec["video_meta"])
    rec["video_meta"]["language"] = "EN"
    post = TikTokPost.model_validate(rec)
    assert post.video_meta.language == "en"


def test_format_validation_summary_and_details():
    text = format_validation_summary(10, 3)
    assert "10" in text and "7 passed" in text and "3 failed" in text
    details = format_failure_details([(1, ["a", "b"]), (5, ["c"])], max_failures=1)
    assert "Line 1:" in details
    assert "Line 5:" not in details
    assert "not shown" in details
