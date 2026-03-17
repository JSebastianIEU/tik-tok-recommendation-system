from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_MODES = {"keyword", "hashtag"}
VALID_COMMENT_SORT = {"top", "new"}


@dataclass(frozen=True)
class PipelineConfig:
    keywords: list[str]
    hashtags: list[str]
    per_query_video_limit: int
    max_comments_per_video: int
    comment_sort: str
    modes_enabled: list[str]
    concurrency: int
    headless: bool
    db_url: str | None
    output_raw_jsonl: bool
    output_raw_jsonl_path: str
    source_label: str
    ms_token: str | None
    tiktok_browser: str


def _read_structured_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - runtime dependency check
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install scraper requirements first."
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported config format for {path}. Use .yaml/.yml or .json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object/map: {path}")
    return data


def _str_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config '{key}' must be a list of strings.")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Config '{key}' must contain only strings.")
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _get_int(data: dict[str, Any], key: str, default: int, *, min_value: int = 0) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Config '{key}' must be an integer.")
    if value < min_value:
        raise ValueError(f"Config '{key}' must be >= {min_value}.")
    return value


def _get_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config '{key}' must be boolean.")
    return value


def _normalize_modes(modes_raw: list[str]) -> list[str]:
    if not modes_raw:
        return ["keyword", "hashtag"]
    out: list[str] = []
    for mode in modes_raw:
        normalized = mode.strip().lower()
        if normalized not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Valid values: {', '.join(sorted(VALID_MODES))}"
            )
        if normalized not in out:
            out.append(normalized)
    return out


def _normalize_hashtags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        cleaned = tag.strip().lstrip("#")
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _parse_output_jsonl(data: dict[str, Any]) -> tuple[bool, str]:
    raw = data.get("output_raw_jsonl", False)
    path_default = str(data.get("output_raw_jsonl_path", "data/raw/pipeline_raw.jsonl"))

    if isinstance(raw, bool):
        return raw, path_default
    if isinstance(raw, dict):
        enabled = raw.get("enabled", False)
        path = raw.get("path", path_default)
        if not isinstance(enabled, bool):
            raise ValueError("Config 'output_raw_jsonl.enabled' must be boolean.")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Config 'output_raw_jsonl.path' must be a non-empty string.")
        return enabled, path

    raise ValueError("Config 'output_raw_jsonl' must be bool or object {enabled, path}.")


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg_path = cfg_path.resolve()

    data = _read_structured_config(cfg_path)

    keywords = _str_list(data.get("keywords"), "keywords")
    hashtags = _normalize_hashtags(_str_list(data.get("hashtags"), "hashtags"))
    modes_enabled = _normalize_modes(_str_list(data.get("modes_enabled"), "modes_enabled"))
    per_query_video_limit = _get_int(
        data,
        "per_query_video_limit",
        default=20,
        min_value=1,
    )
    max_comments_per_video = _get_int(
        data,
        "max_comments_per_video",
        default=200,
        min_value=0,
    )
    concurrency = _get_int(data, "concurrency", default=2, min_value=1)
    headless = _get_bool(data, "headless", default=True)

    comment_sort = str(data.get("comment_sort", "top")).strip().lower()
    if comment_sort not in VALID_COMMENT_SORT:
        raise ValueError(
            f"Config 'comment_sort' must be one of: {', '.join(sorted(VALID_COMMENT_SORT))}"
        )

    output_raw_jsonl, output_raw_jsonl_path = _parse_output_jsonl(data)
    output_path = Path(output_raw_jsonl_path)
    if not output_path.is_absolute():
        output_path = (cfg_path.parent / output_path).resolve()
    output_raw_jsonl_path = str(output_path)

    db_url = data.get("db_url")
    if db_url is not None and not isinstance(db_url, str):
        raise ValueError("Config 'db_url' must be string or null.")

    ms_token = data.get("ms_token")
    if ms_token is not None and not isinstance(ms_token, str):
        raise ValueError("Config 'ms_token' must be string or null.")

    tiktok_browser = str(data.get("tiktok_browser", "chromium")).strip() or "chromium"
    source_label = str(data.get("source_label", "pipeline")).strip() or "pipeline"

    if "keyword" in modes_enabled and not keywords:
        raise ValueError("Mode 'keyword' is enabled but config.keywords is empty.")
    if "hashtag" in modes_enabled and not hashtags:
        raise ValueError("Mode 'hashtag' is enabled but config.hashtags is empty.")

    return PipelineConfig(
        keywords=keywords,
        hashtags=hashtags,
        per_query_video_limit=per_query_video_limit,
        max_comments_per_video=max_comments_per_video,
        comment_sort=comment_sort,
        modes_enabled=modes_enabled,
        concurrency=concurrency,
        headless=headless,
        db_url=db_url,
        output_raw_jsonl=output_raw_jsonl,
        output_raw_jsonl_path=output_raw_jsonl_path,
        source_label=source_label,
        ms_token=ms_token,
        tiktok_browser=tiktok_browser,
    )
