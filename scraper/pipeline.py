from __future__ import annotations

import asyncio
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional
from urllib.parse import quote_plus, urlparse

from selenium.webdriver.common.by import By

from scraper.config import PipelineConfig
from scraper.selenium_utils import dismiss_communication_banner, dismiss_cookie_banner
from scraper.tiktok_post_scraper import create_driver, scrape_tiktok_batch

try:  # pragma: no cover - optional runtime dependency
    from TikTokApi import TikTokApi
except ImportError:  # pragma: no cover - optional runtime dependency
    TikTokApi = None  # type: ignore[assignment]


@dataclass(frozen=True)
class VideoCandidate:
    url: str
    mode: str
    query: str
    position: int
    video_id: Optional[str] = None
    author_unique_id: Optional[str] = None
    plays: Optional[int] = None
    likes: Optional[int] = None


@dataclass
class PipelineSummary:
    scrape_run_id: str
    total_queries: int
    discovered_total: int
    unique_videos: int
    scraped_ok: int
    scraped_failed: int
    persisted_ok: int
    persisted_skipped: int
    persist_failed: int
    comments_written: int
    unique_authors: int
    started_at: datetime
    ended_at: datetime


def _log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {message}")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_video_url(url: str | None) -> str | None:
    if not url:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    if cleaned.startswith("/"):
        cleaned = f"https://www.tiktok.com{cleaned}"
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    if "/video/" not in cleaned:
        return None
    return cleaned.rstrip("/")


def _build_video_url_from_raw(raw: dict[str, Any]) -> str | None:
    share_url = raw.get("shareUrl") or raw.get("share_url") or raw.get("webVideoUrl")
    normalized = _normalize_video_url(share_url if isinstance(share_url, str) else None)
    if normalized:
        return normalized

    video_id = raw.get("id")
    author = raw.get("author") or {}
    unique_id = author.get("uniqueId") if isinstance(author, dict) else None
    if video_id and unique_id:
        return _normalize_video_url(f"https://www.tiktok.com/@{unique_id}/video/{video_id}")
    return None


def _video_metric_tuple(raw: dict[str, Any]) -> tuple[int, int]:
    stats = raw.get("stats") or {}
    if not isinstance(stats, dict):
        return (0, 0)
    plays = _safe_int(stats.get("playCount")) or 0
    likes = _safe_int(stats.get("diggCount")) or _safe_int(stats.get("likeCount")) or 0
    return (plays, likes)


def _rank_raw_videos(raw_videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_metrics = any(_video_metric_tuple(v) != (0, 0) for v in raw_videos)
    if not has_metrics:
        return raw_videos
    return sorted(raw_videos, key=_video_metric_tuple, reverse=True)


def _to_candidates(
    raw_videos: list[dict[str, Any]],
    *,
    mode: str,
    query: str,
    limit: int,
) -> list[VideoCandidate]:
    candidates: list[VideoCandidate] = []
    seen_urls: set[str] = set()
    ranked = _rank_raw_videos(raw_videos)

    for raw in ranked:
        url = _build_video_url_from_raw(raw)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        stats = raw.get("stats") or {}
        author = raw.get("author") or {}
        plays = _safe_int(stats.get("playCount")) if isinstance(stats, dict) else None
        likes = None
        if isinstance(stats, dict):
            likes = _safe_int(stats.get("diggCount")) or _safe_int(stats.get("likeCount"))
        author_uid = author.get("uniqueId") if isinstance(author, dict) else None
        if author_uid is not None and not isinstance(author_uid, str):
            author_uid = str(author_uid)

        candidates.append(
            VideoCandidate(
                url=url,
                mode=mode,
                query=query,
                position=len(candidates) + 1,
                video_id=str(raw.get("id")) if raw.get("id") is not None else None,
                author_unique_id=author_uid,
                plays=plays,
                likes=likes,
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def _parse_proxy_line(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    scheme = "http"
    host_port = line
    if "://" in line:
        parsed = urlparse(line)
        if not parsed.netloc:
            return None
        scheme = parsed.scheme or "http"
        host_port = parsed.netloc

    if ":" not in host_port:
        return None
    host, port_str = host_port.split(":", 1)
    if not host or not port_str:
        return None
    try:
        int(port_str)
    except ValueError:
        return None
    return {"server": f"{scheme}://{host}:{port_str}"}


def _load_random_proxy(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    proxies: list[dict[str, str]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_proxy_line(line)
        if parsed:
            proxies.append(parsed)
    if not proxies:
        return None
    return random.choice(proxies)


def _video_as_dict(video: Any) -> dict[str, Any] | None:
    if isinstance(video, dict):
        return video
    as_dict = getattr(video, "as_dict", None)
    if isinstance(as_dict, dict):
        return as_dict
    if callable(as_dict):
        try:
            maybe = as_dict()
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            return None
    return None


async def _collect_video_dicts(stream: Any, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if hasattr(stream, "__aiter__"):
        async for item in stream:
            payload = _video_as_dict(item)
            if payload:
                out.append(payload)
            if len(out) >= limit:
                break
        return out

    if asyncio.iscoroutine(stream):
        resolved = await stream
        return await _collect_video_dicts(resolved, limit=limit)

    if isinstance(stream, Iterable):
        for item in stream:
            payload = _video_as_dict(item)
            if payload:
                out.append(payload)
            if len(out) >= limit:
                break
        return out

    return out


async def _discover_keyword_with_api(
    api: Any,
    keyword: str,
    limit: int,
) -> list[dict[str, Any]]:
    search = getattr(api, "search", None)
    if search is None:
        raise RuntimeError("TikTokApi search is not available.")

    videos_fn = getattr(search, "videos", None)
    if not callable(videos_fn):
        raise RuntimeError("TikTokApi search.videos is not available.")

    variants = [
        lambda: videos_fn(keyword, count=limit),
        lambda: videos_fn(query=keyword, count=limit),
        lambda: videos_fn(keyword=keyword, count=limit),
        lambda: videos_fn(keywords=keyword, count=limit),
    ]

    last_error: Exception | None = None
    for factory in variants:
        try:
            rows = await _collect_video_dicts(factory(), limit=limit)
            if rows:
                return rows
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(f"TikTokApi keyword search failed for '{keyword}': {last_error}")


async def _discover_query_with_tiktokapi_async(
    *,
    mode: str,
    query: str,
    limit: int,
    ms_token: str,
    browser: str,
    proxies_file: str | None,
) -> list[dict[str, Any]]:
    if TikTokApi is None:
        raise RuntimeError("TikTokApi is not installed.")

    async with TikTokApi() as api:
        attempts = 3
        for attempt in range(1, attempts + 1):
            proxy_cfg = _load_random_proxy(proxies_file)
            try:
                await api.create_sessions(
                    ms_tokens=[ms_token],
                    num_sessions=1,
                    sleep_after=3,
                    browser=browser,
                    proxies=[proxy_cfg] if proxy_cfg else None,
                    timeout=60000,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == attempts:
                    raise RuntimeError(
                        f"Failed to create TikTokApi session after {attempts} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(min(8, 2**attempt))

        target_count = max(limit * 2, limit)
        if mode == "hashtag":
            hashtag = api.hashtag(query.lstrip("#"))
            return await _collect_video_dicts(hashtag.videos(count=target_count), limit=target_count)
        if mode == "keyword":
            return await _discover_keyword_with_api(api, keyword=query, limit=target_count)
        raise ValueError(f"Unsupported mode: {mode}")


def _discover_query_with_tiktokapi(
    *,
    mode: str,
    query: str,
    limit: int,
    ms_token: str,
    browser: str,
    proxies_file: str | None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _discover_query_with_tiktokapi_async(
            mode=mode,
            query=query,
            limit=limit,
            ms_token=ms_token,
            browser=browser,
            proxies_file=proxies_file,
        )
    )


def _discover_query_with_selenium(
    *,
    mode: str,
    query: str,
    limit: int,
    headless: bool,
) -> list[VideoCandidate]:
    driver = create_driver(headless=headless)
    try:
        search_text = query if mode == "keyword" else f"#{query.lstrip('#')}"
        search_url = f"https://www.tiktok.com/search/video?q={quote_plus(search_text)}"
        driver.get(search_url)
        time.sleep(2.0)
        dismiss_communication_banner(driver)
        dismiss_cookie_banner(driver)
        time.sleep(0.5)

        urls: list[str] = []
        seen_urls: set[str] = set()
        idle_rounds = 0
        max_rounds = max(25, limit * 5)

        for _ in range(max_rounds):
            before = len(urls)
            elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/video/']")
            for el in elements:
                href = el.get_attribute("href")
                normalized = _normalize_video_url(href)
                if normalized and normalized not in seen_urls:
                    seen_urls.add(normalized)
                    urls.append(normalized)
                    if len(urls) >= limit:
                        break
            if len(urls) >= limit:
                break

            if len(urls) == before:
                idle_rounds += 1
            else:
                idle_rounds = 0

            if idle_rounds >= 6:
                break

            driver.execute_script("window.scrollBy(0, 1200);")
            time.sleep(1.0)

        candidates: list[VideoCandidate] = []
        for idx, url in enumerate(urls[:limit], start=1):
            candidates.append(
                VideoCandidate(
                    url=url,
                    mode=mode,
                    query=query,
                    position=idx,
                )
            )
        return candidates
    finally:
        driver.quit()


def _discover_query_candidates(
    *,
    mode: str,
    query: str,
    config: PipelineConfig,
) -> list[VideoCandidate]:
    ms_token = config.ms_token or os.getenv("MS_TOKEN")
    raw_videos: list[dict[str, Any]] = []

    if ms_token:
        try:
            _log(f"[discover] {mode}:{query} via TikTokApi")
            raw_videos = _discover_query_with_tiktokapi(
                mode=mode,
                query=query,
                limit=config.per_query_video_limit,
                ms_token=ms_token,
                browser=config.tiktok_browser,
                proxies_file=config.proxies_file,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"[discover] TikTokApi fallback for {mode}:{query} ({exc})")
    else:
        _log(f"[discover] No MS_TOKEN; using Selenium discovery for {mode}:{query}")

    if raw_videos:
        return _to_candidates(
            raw_videos,
            mode=mode,
            query=query,
            limit=config.per_query_video_limit,
        )

    _log(f"[discover] {mode}:{query} via Selenium search fallback")
    return _discover_query_with_selenium(
        mode=mode,
        query=query,
        limit=config.per_query_video_limit,
        headless=config.headless,
    )


def _build_query_jobs(config: PipelineConfig) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    if "keyword" in config.modes_enabled:
        for kw in config.keywords:
            jobs.append(("keyword", kw))
    if "hashtag" in config.modes_enabled:
        for tag in config.hashtags:
            jobs.append(("hashtag", tag))
    return jobs


def _resolve_db_url(config: PipelineConfig, *, db_url_override: str | None = None) -> str | None:
    raw = (db_url_override or os.getenv("DATABASE_URL") or config.db_url or "").strip()
    return raw or None


def _summary_path(config: PipelineConfig, started_at: datetime) -> Path:
    base_dir = Path(config.output_raw_jsonl_path).resolve().parent
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return base_dir / f"pipeline_summary_{stamp}.json"


def _write_summary_file(summary: PipelineSummary, path: Path) -> None:
    payload = {
        "scrape_run_id": summary.scrape_run_id,
        "total_queries": summary.total_queries,
        "discovered_total": summary.discovered_total,
        "unique_videos": summary.unique_videos,
        "scraped_ok": summary.scraped_ok,
        "scraped_failed": summary.scraped_failed,
        "persisted_ok": summary.persisted_ok,
        "persisted_skipped": summary.persisted_skipped,
        "persist_failed": summary.persist_failed,
        "comments_written": summary.comments_written,
        "unique_authors": summary.unique_authors,
        "started_at": summary.started_at.isoformat(),
        "ended_at": summary.ended_at.isoformat(),
        "duration_sec": int((summary.ended_at - summary.started_at).total_seconds()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: PipelineConfig, *, db_url_override: str | None = None) -> PipelineSummary:
    jobs = _build_query_jobs(config)
    if not jobs:
        raise ValueError("No queries to process. Check config keywords/hashtags and modes_enabled.")

    started_at = datetime.now(timezone.utc)
    db_url = _resolve_db_url(config, db_url_override=db_url_override)
    db_enabled = db_url is not None

    scrape_ctx: Any = None
    db_writer = None
    if db_enabled:
        try:
            from scraper.db.writer import BatchedRecordWriter, create_scrape_run
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "DB mode requires psycopg. Install scraper requirements or run with db_url: null."
            ) from exc
        scrape_ctx = create_scrape_run(config.source_label, db_url=db_url)
        commit_every = max(1, int(os.getenv("SCRAPER_DB_COMMIT_EVERY", "50")))
        db_writer = BatchedRecordWriter(db_url=db_url, commit_every=commit_every)
        _log(f"Scrape run created: {scrape_ctx.scrape_run_id} (source={config.source_label})")
    else:
        _log("DB disabled (no db_url/DATABASE_URL). Running in JSON-only mode.")

    discovered_total = 0
    url_to_candidate: dict[str, VideoCandidate] = {}

    for mode, query in jobs:
        candidates = _discover_query_candidates(mode=mode, query=query, config=config)
        discovered_total += len(candidates)
        _log(f"[discover] {mode}:{query} -> {len(candidates)} videos")
        for candidate in candidates:
            if candidate.url not in url_to_candidate:
                url_to_candidate[candidate.url] = candidate

    unique_urls = list(url_to_candidate.keys())
    _log(
        f"[discover] total={discovered_total} | unique={len(unique_urls)} | queries={len(jobs)}"
    )

    raw_path: str | None = None
    raw_output_enabled = config.output_raw_jsonl or not db_enabled
    if raw_output_enabled:
        raw_file = Path(config.output_raw_jsonl_path)
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text("", encoding="utf-8")
        raw_path = str(raw_file)
        if config.output_raw_jsonl:
            _log(f"[output] raw JSONL enabled at {raw_path}")
        else:
            _log(f"[output] JSON-only mode forced raw JSONL output at {raw_path}")

    scraped_ok = 0
    scraped_failed = 0
    persisted_ok = 0
    persisted_skipped = 0
    persist_failed = 0
    comments_written = 0
    unique_authors: set[str] = set()

    writer_ctx = db_writer if db_writer is not None else nullcontext(None)
    with writer_ctx as active_writer:
        if unique_urls:
            for result in scrape_tiktok_batch(
                unique_urls,
                workers=config.concurrency,
                headless=config.headless,
                fast_mode=False,
                max_comments=config.max_comments_per_video,
                comment_sort=config.comment_sort,
                output_jsonl=raw_path,
            ):
                normalized_url = _normalize_video_url(result.get("url"))
                candidate = url_to_candidate.get(normalized_url or "")

                if result.get("success"):
                    scraped_ok += 1
                    normalized = result.get("normalized") or {}
                    if not isinstance(normalized, dict):
                        continue

                    video_block = normalized.get("video")
                    if isinstance(video_block, dict) and candidate:
                        video_block["source"] = f"{candidate.mode}:{candidate.query}"
                        normalized["video"] = video_block

                    comments = normalized.get("comments") or []
                    if isinstance(comments, list):
                        comments_written += len(comments)
                    author = normalized.get("author") or {}
                    if isinstance(author, dict):
                        author_id = author.get("author_id")
                        if isinstance(author_id, str) and author_id:
                            unique_authors.add(author_id)

                    if db_enabled and active_writer is not None:
                        try:
                            wrote = active_writer.write(
                                normalized,
                                scrape_ctx=scrape_ctx,
                                position=candidate.position if candidate else None,
                                skip_existing=True,
                            )
                            if wrote:
                                persisted_ok += 1
                            else:
                                persisted_skipped += 1
                        except Exception as exc:  # noqa: BLE001
                            persist_failed += 1
                            active_writer.rollback()
                            _log(f"[persist] failed for {result.get('url')}: {exc}")
                    else:
                        persisted_ok += 1
                else:
                    scraped_failed += 1
                    _log(f"[scrape] failed {result.get('url')}: {result.get('error')}")

    ended_at = datetime.now(timezone.utc)
    summary = PipelineSummary(
        scrape_run_id=str(scrape_ctx.scrape_run_id) if scrape_ctx else "json_only",
        total_queries=len(jobs),
        discovered_total=discovered_total,
        unique_videos=len(unique_urls),
        scraped_ok=scraped_ok,
        scraped_failed=scraped_failed,
        persisted_ok=persisted_ok,
        persisted_skipped=persisted_skipped,
        persist_failed=persist_failed,
        comments_written=comments_written,
        unique_authors=len(unique_authors),
        started_at=started_at,
        ended_at=ended_at,
    )
    summary_file = _summary_path(config, started_at)
    _write_summary_file(summary, summary_file)
    _log(f"[summary] wrote {summary_file}")
    return summary
