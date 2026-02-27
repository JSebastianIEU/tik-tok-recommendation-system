"""
Selenium-based TikTok post scraper.

This is an adapted version of the provided reference implementation,
augmented with:
- Shadow DOM–aware cookie banner handling.
- Optional persistence into the Postgres 3NF schema via scraper.db.writer.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.request import urlretrieve

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraper.db.writer import ScrapeContext, create_scrape_run, dry_run_print_sql, write_normalized_record
from scraper.selenium_utils import dismiss_communication_banner, dismiss_cookie_banner


def create_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1280,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def load_tiktok_video_page(
    driver: webdriver.Chrome,
    video_url: str,
    wait_seconds: int = 8,
    post_load_sleep: float = 2.0,
) -> str:
    driver.get(video_url)
    WebDriverWait(driver, wait_seconds).until(EC.presence_of_element_located((By.TAG_NAME, "script")))
    time.sleep(post_load_sleep)
    return driver.page_source


def extract_rehydration_json(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__UNIVERSAL_DATA_FOR_REHYDRATION__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


def _find_in_dict(obj: Any, key: str, results: Optional[list] = None) -> list:
    if results is None:
        results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                results.append(v)
            _find_in_dict(v, key, results)
    elif isinstance(obj, list):
        for item in obj:
            _find_in_dict(item, key, results)
    return results


def _get_video_item_payload(data: dict) -> Optional[dict]:
    for key in ("defaultProps", "props", "pageProps"):
        if key in data and isinstance(data[key], dict):
            payload = data[key]
            for scope in ("__DEFAULT_SCOPE__", "webapp.video-detail", "videoDetail"):
                if scope in payload:
                    scope_data = payload[scope]
                    if isinstance(scope_data, dict) and "itemInfo" in scope_data:
                        return scope_data
                    if isinstance(scope_data, dict) and "itemList" in scope_data:
                        items = scope_data["itemList"]
                        if items and isinstance(items[0], dict):
                            return {"itemInfo": {"itemStruct": items[0]}, "itemList": items}
            if "itemInfo" in payload:
                return payload
    item_infos = _find_in_dict(data, "itemInfo")
    for info in item_infos:
        if isinstance(info, dict) and "itemStruct" in info:
            return {"itemInfo": info}
    return None


def _get_comments_from_data(data: dict) -> list[dict]:
    comments = _find_in_dict(data, "comments")
    for c in comments:
        if isinstance(c, list) and len(c) > 0:
            return c
    for key in ("commentList", "comments", "itemList"):
        found = _find_in_dict(data, key)
        for f in found:
            if isinstance(f, list):
                for item in f:
                    if isinstance(item, dict) and ("text" in item or "comment" in item):
                        return f
    return []


def _click_comment_icon(driver: webdriver.Chrome, timeout: int = 8) -> None:
    try:
        button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[starts-with(@aria-label,'Read or add comments')]"
                    " | //button[.//span[@data-e2e='comment-icon']]",
                )
            )
        )
        try:
            button.click()
        except Exception:
            driver.execute_script("arguments[0].click();", button)

        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='DivCommentListContainer']")
            )
        )
        time.sleep(0.5)
    except Exception:
        return


def _scroll_to_load_comments(driver: webdriver.Chrome, wait_seconds: int = 6) -> None:
    try:
        dismiss_communication_banner(driver)
        dismiss_cookie_banner(driver)
        _click_comment_icon(driver)

        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(1.5)
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(1.5)
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(1)
        try:
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "div[class*='DivCommentListContainer'], "
                        "div[class*='DivCommentObjectWrapper'], "
                        "div[class*='DivCommentItemWrapper']",
                    )
                )
            )
        except Exception:
            pass
        time.sleep(1.5)
    except Exception:
        return


def _get_comments_from_dom(driver: webdriver.Chrome, max_comments: int) -> list[dict]:
    _scroll_to_load_comments(driver, wait_seconds=4)

    comments: list[dict] = []
    seen_texts: set[str] = set()

    def _parse_count(num_text: str) -> int:
        if not num_text:
            return 0
        text = num_text.replace(",", "").strip()
        try:
            import re as _re

            m = _re.match(r"([\d\.]+)\s*([KkMm])?", text)
            if not m:
                return 0
            value = float(m.group(1))
            suffix = m.group(2)
            if suffix in ("K", "k"):
                value *= 1_000
            elif suffix in ("M", "m"):
                value *= 1_000_000
            return int(value)
        except Exception:
            return 0

    def _add_comment(text: str, like_count: int = 0, reply_count: int = 0) -> bool:
        text = (text or "").strip()
        if not text or len(text) > 2000 or text in seen_texts:
            return False
        seen_texts.add(text)
        comments.append(
            {
                "text": text,
                "likeCount": like_count,
                "replyCount": reply_count,
                "from_dom": True,
            }
        )
        return True

    selectors = [
        "div[class*='DivCommentObjectWrapper']",
        "div[class*='DivCommentItemWrapper']",
        "div[class*='DivCommentListContainer'] > div",
    ]

    for selector in selectors:
        if len(comments) >= max_comments:
            break
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements[: max_comments * 2]:
                if len(comments) >= max_comments:
                    break
                try:
                    text_el = None
                    try:
                        level1 = el.find_elements(By.CSS_SELECTOR, "[data-e2e='comment-level-1']")
                        if level1:
                            text_el = level1[0]
                    except Exception:
                        text_el = None

                    if not text_el:
                        for text_sel in ("span", "p"):
                            try:
                                text_els = el.find_elements(By.CSS_SELECTOR, text_sel)
                                if text_els:
                                    text_el = text_els[0]
                                    break
                            except Exception:
                                continue

                    if not text_el:
                        continue

                    text = (text_el.text or "").strip()
                    if not (5 < len(text) < 1500):
                        continue

                    like_count = 0
                    try:
                        like_span = el.find_elements(
                            By.CSS_SELECTOR, "div[class*='DivLikeContainer'] span"
                        )
                        if like_span:
                            like_count = _parse_count(like_span[0].text)
                    except Exception:
                        like_count = 0

                    reply_count = 0
                    try:
                        reply_label_els = el.find_elements(
                            By.XPATH,
                            "following-sibling::div[contains(@class,'DivReplyContainer')][1]"
                            "//div[contains(@class,'TUXButton-label')]",
                        )
                        if reply_label_els:
                            label = (reply_label_els[0].text or "").strip()
                            import re as _re2

                            m = _re2.search(r"View\s+(\d+)\s+repl", label)
                            if m:
                                reply_count = int(m.group(1))
                    except Exception:
                        reply_count = 0

                    _add_comment(text, like_count=like_count, reply_count=reply_count)
                except Exception:
                    continue
            if comments:
                break
        except Exception:
            continue

    if not comments:
        try:
            level1_els = driver.find_elements(By.CSS_SELECTOR, "[data-e2e='comment-level-1']")
            for el in level1_els[: max_comments * 2]:
                if len(comments) >= max_comments:
                    break
                try:
                    t = (el.text or "").strip()
                    if 5 < len(t) < 1500:
                        _add_comment(t)
                except Exception:
                    continue
        except Exception:
            pass

    return comments[:max_comments]


def _safe_get(obj: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def parse_video_item(item: dict) -> dict:
    author = item.get("author", {}) or {}
    music = item.get("music", {}) or {}
    stats = item.get("stats", {}) or {}
    author_stats = item.get("authorStats", {}) or {}
    video = item.get("video", {}) or {}

    hashtags: list[str] = []
    challenges = item.get("challenges") or []
    if isinstance(challenges, list):
        for ch in challenges:
            if not isinstance(ch, dict):
                continue
            tag = ch.get("title") or ch.get("titleAlias") or ch.get("chaName") or ch.get("name")
            if tag and isinstance(tag, str):
                hashtags.append(tag)

    play_addr = video.get("playAddr") or video.get("playaddr") or ""
    if not play_addr and "playApi" in video:
        play_addr = video["playApi"]
    download_addr = video.get("downloadAddr") or video.get("downloadaddr") or play_addr

    music_id = music.get("id")
    music_title = music.get("title") or music.get("titleName") or ""
    music_author = music.get("authorName") or music.get("ownerNickname") or ""
    music_url = music.get("playUrl") or music.get("play") or ""

    share_url = item.get("shareUrl") or item.get("shareUrlNew") or ""
    is_ad = bool(item.get("isAd"))
    region = item.get("region") or item.get("locationCreated") or ""

    return {
        "id": item.get("id"),
        "caption": (item.get("desc") or "").strip(),
        "createTime": item.get("createTime"),
        "author": {
            "id": author.get("id"),
            "uniqueId": author.get("uniqueId"),
            "nickname": author.get("nickname"),
            "verified": author.get("verified"),
            "avatarThumb": author.get("avatarThumb"),
            "avatarMedium": author.get("avatarMedium"),
            "avatarLarger": author.get("avatarLarger"),
        },
        "authorStats": {
            "followingCount": author_stats.get("followingCount"),
            "followerCount": author_stats.get("followerCount"),
            "heartCount": author_stats.get("heart") or author_stats.get("heartCount"),
            "videoCount": author_stats.get("videoCount"),
            "diggCount": author_stats.get("diggCount"),
        },
        "stats": {
            "playCount": stats.get("playCount"),
            "likeCount": stats.get("diggCount") or stats.get("likeCount"),
            "commentCount": stats.get("commentCount"),
            "shareCount": stats.get("shareCount"),
        },
        "video": {
            "playUrl": play_addr,
            "downloadUrl": download_addr,
            "duration": video.get("duration"),
            "width": video.get("width"),
            "height": video.get("height"),
            "cover": video.get("cover")
            or video.get("originCover")
            or video.get("dynamicCover"),
            "ratio": video.get("ratio"),
            "fps": video.get("fps"),
        },
        "music": {
            "id": music_id,
            "title": music_title,
            "authorName": music_author,
            "playUrl": music_url,
            "isOriginalSound": music.get("original") or music.get("isOriginalSound"),
            "album": music.get("album"),
            "coverThumb": music.get("coverThumb")
            or music.get("coverMedium")
            or music.get("coverLarge"),
        },
        "hashtags": hashtags,
        "shareUrl": share_url,
        "isAd": is_ad,
        "region": region,
        "direct_url": item.get("webVideoUrl") or item.get("video", {}).get("playAddr"),
    }


def parse_comment(raw: dict) -> dict:
    text = raw.get("text") or _safe_get(raw, "comment", "text") or ""
    if isinstance(text, dict):
        text = text.get("text", "")
    user = raw.get("user") or _safe_get(raw, "user")
    author_id = None
    username = None
    if isinstance(user, dict):
        author_id = user.get("id") or user.get("uid")
        username = user.get("uniqueId") or user.get("nickname")
    return {
        "id": raw.get("id") or raw.get("cid"),
        "text": text,
        "author_id": author_id,
        "username": username,
        "likeCount": raw.get("diggCount") or raw.get("likeCount", 0),
        "replyCount": raw.get("replyCommentTotal", 0),
        "createTime": raw.get("createTime"),
    }


def scrape_tiktok_post(
    video_url: str,
    *,
    driver: Optional[webdriver.Chrome] = None,
    headless: bool = True,
    max_comments: int = 10,
    fast_mode: bool = False,
    download_video_path: Optional[str] = None,
    download_audio_path: Optional[str] = None,
) -> dict:
    own_driver = driver is None
    if driver is None:
        driver = create_driver(headless=headless)

    try:
        want_comments = max(0, min(max_comments, 5))
        page_wait = 5 if fast_mode else 8
        post_sleep = 1.0 if fast_mode else 2.0

        html = load_tiktok_video_page(
            driver, video_url, wait_seconds=page_wait, post_load_sleep=post_sleep
        )
        data = extract_rehydration_json(html)
        if not data:
            return {
                "success": False,
                "error": "Could not find __UNIVERSAL_DATA_FOR_REHYDRATION__ on page",
            }

        payload = _get_video_item_payload(data)
        if not payload:
            return {
                "success": False,
                "error": "Could not find video item in page data",
            }

        item_info = payload.get("itemInfo", {})
        item = item_info.get("itemStruct") if item_info else None
        if not item and payload.get("itemList"):
            item = payload["itemList"][0]

        if not item:
            return {
                "success": False,
                "error": "Video item structure not found",
            }

        parsed = parse_video_item(item)
        scraped_at = datetime.now(timezone.utc).isoformat()
        top_comments_raw = _get_comments_from_data(data)[:want_comments] if want_comments else []
        top_comments = [parse_comment(c) for c in top_comments_raw if isinstance(c, dict)]
        if want_comments and len(top_comments) < want_comments and not fast_mode:
            dom_comments = _get_comments_from_dom(driver, want_comments - len(top_comments))
            for c in dom_comments:
                if len(top_comments) >= want_comments:
                    break
                top_comments.append(
                    {
                        "id": c.get("id"),
                        "text": c.get("text", ""),
                        "likeCount": c.get("likeCount", 0),
                        "replyCount": c.get("replyCount", 0),
                        "createTime": c.get("createTime"),
                        "from_dom": True,
                    }
                )

        author_id = parsed["author"].get("id")
        video_id = parsed.get("id")

        normalized_author = {
            "author_id": author_id,
            "username": parsed["author"].get("uniqueId"),
            "display_name": parsed["author"].get("nickname"),
            "bio": item.get("author", {}).get("signature"),
            "avatar_url": parsed["author"].get("avatarLarger")
            or parsed["author"].get("avatarMedium")
            or parsed["author"].get("avatarThumb"),
            "verified": bool(parsed["author"].get("verified")),
        }

        audio_name = parsed["music"].get("title") or ""
        if parsed["music"].get("isOriginalSound"):
            owner = (
                parsed["music"].get("authorName")
                or parsed["author"].get("nickname")
                or parsed["author"].get("uniqueId")
            )
            audio_name = f"Original sound - {owner}" if owner else "Original sound"

        normalized_video = {
            "video_id": video_id,
            "author_id": author_id,
            "scraped_at": scraped_at,
            "url": video_url,
            "caption": parsed.get("caption"),
            "hashtags": parsed.get("hashtags", []),
            "audio_name": audio_name,
            "audio_id": parsed["music"].get("id"),
            "duration_sec": parsed["video"].get("duration"),
            "thumbnail_url": parsed["video"].get("cover"),
            "created_at": str(parsed.get("createTime"))
            if parsed.get("createTime") is not None
            else None,
            "likes": parsed["stats"].get("likeCount"),
            "comments_count": parsed["stats"].get("commentCount"),
            "shares": parsed["stats"].get("shareCount"),
            "plays": parsed["stats"].get("playCount"),
            "source": None,
            "position": None,
        }

        normalized_author_metrics = {
            "author_id": author_id,
            "video_id": video_id,
            "scraped_at": scraped_at,
            "follower_count": parsed.get("authorStats", {}).get("followerCount"),
            "following_count": parsed.get("authorStats", {}).get("followingCount"),
            "author_likes_count": parsed.get("authorStats", {}).get("heartCount"),
        }

        normalized_comments = []
        for c in top_comments:
            normalized_comments.append(
                {
                    "comment_id": c.get("id"),
                    "video_id": video_id,
                    "author_id": c.get("author_id"),
                    "username": c.get("username"),
                    "text": c.get("text", ""),
                    "likes": c.get("likeCount", 0),
                    "reply_count": c.get("replyCount", 0),
                    "parent_comment_id": None,
                    "scraped_at": scraped_at,
                }
            )

        result = {
            "success": True,
            "url": video_url,
            "caption": parsed["caption"],
            "metadata": {
                "id": parsed["id"],
                "createTime": parsed["createTime"],
                "author": parsed["author"],
                "authorStats": parsed.get("authorStats"),
                "stats": parsed["stats"],
                "hashtags": parsed.get("hashtags", []),
                "shareUrl": parsed.get("shareUrl"),
                "isAd": parsed.get("isAd"),
                "region": parsed.get("region"),
            },
            "video": parsed["video"],
            "music": parsed["music"],
            "top_comments": top_comments,
            "normalized": {
                "author": normalized_author,
                "video": normalized_video,
                "authorMetricSnapshot": normalized_author_metrics,
                "comments": normalized_comments,
            },
        }

        if download_video_path and parsed["video"].get("playUrl"):
            try:
                urlretrieve(parsed["video"]["playUrl"], download_video_path)
                result["video_downloaded"] = download_video_path
            except Exception as e:
                result["video_download_error"] = str(e)

        if download_audio_path and parsed["music"].get("playUrl"):
            try:
                urlretrieve(parsed["music"]["playUrl"], download_audio_path)
                result["audio_downloaded"] = download_audio_path
            except Exception as e:
                result["audio_download_error"] = str(e)

        return result

    finally:
        if own_driver and driver:
            driver.quit()


def _video_url_from_api_jsonl_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        vid = obj.get("id")
        author = obj.get("author") or {}
        unique_id = author.get("uniqueId") if isinstance(author, dict) else None
        if vid and unique_id:
            return f"https://www.tiktok.com/@{unique_id}/video/{vid}"
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def _read_urls_from_file(path: Path, jsonl_format: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    urls: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if jsonl_format:
            url = _video_url_from_api_jsonl_line(line)
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        else:
            if line.startswith("http") and line not in seen:
                seen.add(line)
                urls.append(line)
    return urls


def _worker_scrape_chunk(
    urls: list[str],
    headless: bool,
    fast_mode: bool,
    max_comments: int,
) -> list[dict]:
    driver = create_driver(headless=headless)
    results: list[dict] = []
    try:
        for url in urls:
            try:
                out = scrape_tiktok_post(
                    url,
                    driver=driver,
                    headless=headless,
                    max_comments=max_comments,
                    fast_mode=fast_mode,
                )
                results.append(out)
            except Exception as e:
                results.append({"success": False, "url": url, "error": str(e)})
    finally:
        driver.quit()
    return results


def scrape_tiktok_batch(
    urls: list[str],
    *,
    workers: int = 4,
    headless: bool = True,
    fast_mode: bool = True,
    max_comments: int = 0,
    output_jsonl: Optional[str] = None,
) -> Iterator[dict]:
    if not urls:
        return
    chunk_size = max(1, (len(urls) + workers - 1) // workers)
    chunks: list[list[str]] = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i : i + chunk_size])
    file_lock = threading.Lock() if output_jsonl else None
    out_path = Path(output_jsonl) if output_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_result(obj: dict) -> None:
        if not out_path or not file_lock:
            return
        with file_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _worker_scrape_chunk,
                chunk,
                headless,
                fast_mode,
                max_comments,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            for result in future.result():
                if output_jsonl:
                    _write_result(result)
                yield result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape TikTok post(s): single URL or batch (many URLs in parallel) "
        "and optionally write into Postgres."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Single TikTok video URL (omit for batch mode)",
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="Show browser window(s)"
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=5,
        help="Max top comments per video (default 5). Use 0 in batch for speed.",
    )
    parser.add_argument(
        "--download-video",
        metavar="PATH",
        help="Download video to path (single-URL only)",
    )
    parser.add_argument(
        "--download-audio",
        metavar="PATH",
        help="Download audio/music to path (single-URL only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output: JSON file (single URL) or JSONL file (batch)",
    )
    parser.add_argument(
        "--urls-file",
        metavar="FILE",
        help="Batch: file with one TikTok URL per line",
    )
    parser.add_argument(
        "--jsonl-file",
        metavar="FILE",
        help="Batch: TikTokApi-style JSONL (id + author.uniqueId); URLs are built automatically",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Batch: number of parallel browser instances (default 4)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Batch: fast mode (shorter waits, no comment DOM scraping)",
    )
    parser.add_argument(
        "--db-url",
        help="Optional Postgres URL (falls back to DATABASE_URL env var)",
    )
    parser.add_argument(
        "--source",
        default="single_url",
        help="Logical source label for this scrape run (e.g. for_you, single_url, manual).",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help="Optional position in a feed run (1-based). Used only in single-URL mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to Postgres; print a summary of what would be written.",
    )
    args = parser.parse_args(argv)

    urls_file = getattr(args, "urls_file", None) or None
    jsonl_file = getattr(args, "jsonl_file", None) or None
    batch_mode = bool(urls_file or jsonl_file)

    if batch_mode:
        if urls_file and jsonl_file:
            print("Use only one of --urls-file or --jsonl-file.", file=sys.stderr)
            return 1
        path = Path(urls_file or jsonl_file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        urls = _read_urls_from_file(path, jsonl_format=bool(jsonl_file))
        if not urls:
            print("No valid URLs found in file.", file=sys.stderr)
            return 1

        out_path = args.output or "tiktok_batch_output.jsonl"
        count = 0
        scrape_ctx: Optional[ScrapeContext] = None
        if not args.dry_run:
            scrape_ctx = create_scrape_run(args.source, db_url=args.db_url)

        for result in scrape_tiktok_batch(
            urls,
            workers=args.workers,
            headless=not args.no_headless,
            fast_mode=args.fast,
            max_comments=args.comments,
            output_jsonl=out_path,
        ):
            count += 1
            status = "ok" if result.get("success") else "fail"
            print(f"[{count}/{len(urls)}] {status} {result.get('url', '')[:60]}...")
            if result.get("success") and "normalized" in result:
                norm = result["normalized"]
                if args.dry_run:
                    dry_run_print_sql(norm)
                else:
                    write_normalized_record(
                        norm,
                        db_url=args.db_url,
                        scrape_ctx=scrape_ctx,
                        position=None,
                    )
        print(f"Done. Wrote {count} records to {out_path}")
        return 0

    if not args.url:
        parser.print_help()
        print(
            "\nProvide a single URL or use --urls-file / --jsonl-file for batch.",
            file=sys.stderr,
        )
        return 1

    result = scrape_tiktok_post(
        args.url,
        headless=not args.no_headless,
        max_comments=args.comments,
        download_video_path=args.download_video,
        download_audio_path=args.download_audio,
    )
    if not result.get("success"):
        print(json.dumps(result, indent=2, default=str))
        return 2

    norm = result.get("normalized")
    if norm:
        if args.dry_run:
            dry_run_print_sql(norm)
        else:
            scrape_ctx = create_scrape_run(args.source, db_url=args.db_url)
            write_normalized_record(
                norm,
                db_url=args.db_url,
                scrape_ctx=scrape_ctx,
                position=args.position,
            )

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(f"Wrote result to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

