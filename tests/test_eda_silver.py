from __future__ import annotations

from eda.src.silver import build_silver_dataset


def test_silver_videos_dedupes_and_derives_rate():
    rows = [
        {
            "video_id": "v1",
            "created_at": "2026-03-10T00:00:00Z",
            "video_author_id": "a1",
            "likes": "10",
            "comments_count": "5",
            "shares": "5",
            "plays": "100",
            "hashtags": ["fitness", "fitness", " workout "],
        },
        {
            "video_id": "v1",
            "created_at": "2026-03-09T00:00:00Z",
            "video_author_id": "a1",
            "likes": "1",
            "comments_count": "1",
            "shares": "1",
            "plays": "10",
        },
    ]

    out = build_silver_dataset("videos", rows)
    assert len(out) == 1
    assert out[0]["likes"] == 10
    assert out[0]["plays"] == 100
    assert abs(out[0]["engagement_rate"] - 0.2) < 1e-9
    assert out[0]["hashtags"] == ["fitness", "workout"]


def test_silver_comments_dedupes_and_adds_flags():
    rows = [
        {
            "comment_id": "c1",
            "video_id": "v1",
            "text": " hello   world ",
            "parent_comment_id": None,
            "row_created_at": "2026-03-10T00:00:00Z",
            "comment_likes": "7",
        },
        {
            "comment_id": "c1",
            "video_id": "v1",
            "text": " hello   world ",
            "parent_comment_id": "p1",
            "row_created_at": "2026-03-11T00:00:00Z",
            "comment_likes": "8",
        },
    ]

    out = build_silver_dataset("comments", rows)
    assert len(out) == 1
    assert out[0]["is_reply"] is True
    assert out[0]["text"] == "hello world"
    assert out[0]["text_length"] == 11
    assert out[0]["comment_likes"] == 8


def test_silver_authors_derives_comments_per_video():
    rows = [
        {
            "author_id": "a1",
            "row_created_at": "2026-03-11T00:00:00Z",
            "videos_count": "4",
            "comments_count": "10",
            "avg_video_plays": "42.5",
        }
    ]

    out = build_silver_dataset("authors", rows)
    assert len(out) == 1
    assert out[0]["videos_count"] == 4
    assert out[0]["comments_count"] == 10
    assert abs(out[0]["comments_per_video"] - 2.5) < 1e-9
    assert abs(out[0]["avg_video_plays"] - 42.5) < 1e-9


def test_silver_full_normalizes_nested_comments():
    rows = [
        {
            "video_id": "v1",
            "created_at": "2026-03-11T00:00:00Z",
            "likes": "2",
            "comments_count": "1",
            "shares": "1",
            "plays": "10",
            "comments": [
                {
                    "comment_id": "c1",
                    "text": "  hey  there  ",
                    "parent_comment_id": None,
                    "likes": "4",
                    "reply_count": "1",
                    "scraped_at": "2026-03-11T00:01:00Z",
                }
            ],
        }
    ]

    out = build_silver_dataset("full", rows)
    assert len(out) == 1
    assert out[0]["comment_count_extracted"] == 1
    assert out[0]["comments"][0]["text"] == "hey there"
    assert out[0]["comments"][0]["text_length"] == 9
    assert out[0]["comments"][0]["is_reply"] is False


def test_silver_invalid_dataset_raises():
    try:
        build_silver_dataset("unknown", [])
        raised = False
    except ValueError:
        raised = True
    assert raised is True
