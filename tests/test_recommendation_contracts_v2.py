from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.recommendation import (
    CONTRACT_VERSION,
    build_contract_manifest,
    load_bundle_from_manifest,
    validate_raw_dataset_jsonl_against_contract,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_contract_v2_assigns_watermarks_and_lateness() -> None:
    raw = "\n".join(
        [
            '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,"views":100,"scraped_at":"2026-01-05T10:00:00Z","author":{"author_id":"ax"}}',
            '{"video_id":"x2","caption":"B","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":3,"shares":1,"views":110,"scraped_at":"2026-01-03T10:00:00Z","author":{"author_id":"ay"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is True
    assert result.bundle is not None
    assert result.bundle.version == CONTRACT_VERSION
    classes = [row.lateness_class for row in result.bundle.video_snapshots]
    assert "on_time" in classes
    assert "late_within_grace" in classes
    assert len(result.bundle.source_watermarks) == 1
    assert result.bundle.source_watermarks[0].records_seen >= 2


def test_contract_v2_conflict_routes_to_quarantine() -> None:
    raw = "\n".join(
        [
            '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}',
            '{"video_id":"x1","caption":"B","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":3,"shares":1,"views":110,"scraped_at":"2026-01-02T10:00:00Z","author":{"author_id":"ay"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is False
    assert any("conflicting author_id" in err for err in result.errors)


def test_contract_v2_late_out_of_watermark_is_quarantined_and_dropped() -> None:
    raw = "\n".join(
        [
            '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,"views":100,"scraped_at":"2026-01-10T10:00:00Z","author":{"author_id":"ax"}}',
            '{"video_id":"x1","caption":"A2","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":3,"shares":1,"views":101,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is True
    assert result.bundle is not None
    assert len(result.bundle.video_snapshots) == 1
    assert len(result.bundle.quarantine_records) >= 1
    assert any(item.reason == "late_out_of_watermark" for item in result.bundle.quarantine_records)


def test_contract_manifest_replay_roundtrip(tmp_path: Path) -> None:
    raw = (
        '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],'
        '"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,'
        '"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}'
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is True
    assert result.bundle is not None
    manifest = build_contract_manifest(
        bundle=result.bundle,
        manifest_root=tmp_path,
        source_file_hashes={"unit": "abc123"},
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    manifest_file = Path(str(manifest["manifest_dir"])) / "manifest.json"
    assert manifest_file.exists()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert payload["manifest_id"] == manifest["manifest_id"]
    restored = load_bundle_from_manifest(manifest_file)
    assert restored.manifest_id == manifest["manifest_id"]
    assert len(restored.videos) == 1


def test_contract_manifest_id_is_content_addressed_stable(tmp_path: Path) -> None:
    raw = (
        '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],'
        '"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,'
        '"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}'
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok and result.bundle is not None
    one = build_contract_manifest(
        bundle=result.bundle,
        manifest_root=tmp_path,
        source_file_hashes={"source": "h1"},
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    two = build_contract_manifest(
        bundle=result.bundle,
        manifest_root=tmp_path,
        source_file_hashes={"source": "h1"},
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert one["manifest_id"] == two["manifest_id"]


def test_manifest_load_fails_on_entity_tamper(tmp_path: Path) -> None:
    raw = (
        '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],'
        '"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,'
        '"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}'
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok and result.bundle is not None
    manifest = build_contract_manifest(
        bundle=result.bundle,
        manifest_root=tmp_path,
        source_file_hashes={"source": "h1"},
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    bundle_file = Path(str(manifest["bundle_file"]))
    payload = json.loads(bundle_file.read_text(encoding="utf-8"))
    payload["videos"][0]["caption"] = "tampered-caption"
    bundle_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        load_bundle_from_manifest(Path(str(manifest["manifest_dir"])) / "manifest.json")
    except ValueError as exc:
        assert "entity hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError due to manifest/entity tamper")
