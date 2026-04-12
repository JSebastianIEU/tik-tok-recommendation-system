"""Deterministic synthetic TikTok data generator.

Produces JSONL records conforming to ``src.common.schemas.TikTokPost``.
All randomness is seeded so output is fully reproducible for CI and tests.

Usage (CLI):
    python -m src.data.mock_generator --count 100 --seed 42
    python -m src.data.mock_generator --count 50 --output data/mock/fresh.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.common.schemas import TikTokPost

# ── content pools (kept small; variety comes from random combination) ──────

_NICHES = [
    ("fitness", ["#fitness", "#workout", "#health"], ["home exercise", "gym routine"]),
    ("food", ["#food", "#recipe", "#cooking"], ["easy recipe", "budget meals"]),
    ("travel", ["#travel", "#explore", "#adventure"], ["travel tips", "itinerary"]),
    ("tech", ["#tech", "#coding", "#webdev"], ["tech tutorial", "programming"]),
    ("beauty", ["#beauty", "#makeup", "#skincare"], ["beauty tips", "skincare routine"]),
    ("music", ["#music", "#piano", "#guitar"], ["learn instrument", "music theory"]),
    ("diy", ["#diy", "#home", "#upcycle"], ["diy project", "home improvement"]),
    ("science", ["#science", "#education", "#learn"], ["science explainer", "experiments"]),
    ("fashion", ["#fashion", "#style", "#ootd"], ["outfit ideas", "thrift haul"]),
    ("gaming", ["#gaming", "#speedrun", "#esports"], ["game tips", "speedrun guide"]),
    ("pets", ["#pets", "#dogs", "#cats"], ["pet training", "animal care"]),
    ("parenting", ["#parenting", "#kids", "#family"], ["parenting tips", "family life"]),
    ("finance", ["#finance", "#investing", "#money"], ["money tips", "investing basics"]),
    ("photography", ["#photography", "#camera", "#tips"], ["photo editing", "camera gear"]),
    ("dance", ["#dance", "#salsa", "#hiphop"], ["dance tutorial", "choreography"]),
]

_CAPTION_TEMPLATES = [
    "{adj} {topic} in {n} steps",
    "How to {verb} like a pro",
    "Quick {topic} for beginners",
    "{n}-minute {topic} routine",
    "Why {topic} matters — simple",
    "My favorite {topic} hack",
    "The truth about {topic}",
    "{topic} tips nobody tells you",
    "I tried {topic} for 30 days",
    "Stop making this {topic} mistake",
]

_ADJECTIVES = ["easy", "quick", "simple", "ultimate", "5-minute", "beginner", "advanced", "daily"]
_VERBS = ["cook", "code", "train", "build", "create", "style", "paint", "edit", "run", "play"]
_LANGUAGES = ["en", "es", "ja", "de", "fr", "pt", "ko", "zh"]

_COMMENT_TEMPLATES = [
    "This is so helpful!",
    "Can you do a follow-up?",
    "Tried this and it works!",
    "Where do I get that?",
    "Saved for later.",
    "Love the energy.",
    "Best tip I've seen today.",
    "More like this please.",
    "What equipment did you use?",
    "Great pacing.",
    "Thanks for sharing.",
    "Amazing results!",
    "Does this work for beginners?",
    "Just subscribed.",
    "Need a part 2!",
    "So satisfying to watch.",
    "Perfect tutorial.",
    "How long did that take?",
    "Any alternatives?",
    "This changed everything for me.",
]

_AUDIO_TITLES = [
    "Upbeat Mix", "Lo-fi Chill", "Ambient Tech", "Soft Keys",
    "Urban Groove", "Latin Rhythm", "Calm Strings", "Retro Pop",
    "Morning Beat", "Synth Wave", "Indie Folk", "Café Background",
]


# ── generator ──────────────────────────────────────────────────────────────

def _rand_username(rng: random.Random) -> str:
    base = rng.choice(["creator", "maker", "pro", "guru", "fan", "life", "tips", "daily"])
    suffix = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 5)))
    return f"{base}_{suffix}"


def _rand_caption(rng: random.Random, topic: str) -> str:
    template = rng.choice(_CAPTION_TEMPLATES)
    return template.format(
        adj=rng.choice(_ADJECTIVES),
        topic=topic,
        verb=rng.choice(_VERBS),
        n=rng.randint(3, 10),
    )


def _rand_datetime(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = end - start
    offset = rng.random() * delta.total_seconds()
    return start + timedelta(seconds=offset)


def generate_post(rng: random.Random, index: int, time_range: tuple[datetime, datetime]) -> Dict[str, Any]:
    """Generate a single synthetic TikTokPost dict."""
    niche_name, hashtags, keywords = rng.choice(_NICHES)
    posted_at = _rand_datetime(rng, time_range[0], time_range[1])
    username = _rand_username(rng)
    num_comments = rng.randint(3, 12)

    views = rng.randint(5_000, 2_000_000)
    likes = rng.randint(int(views * 0.01), int(views * 0.15))
    shares = rng.randint(int(likes * 0.01), int(likes * 0.2))

    comments: List[Dict[str, Any]] = []
    for ci in range(num_comments):
        comment_time = posted_at + timedelta(minutes=rng.randint(5, 1440))
        comments.append({
            "comment_id": f"c{index:04d}_{ci:02d}",
            "text": rng.choice(_COMMENT_TEMPLATES),
            "likes": rng.randint(0, max(1, likes // 10)),
            "created_at": comment_time.isoformat(),
        })

    selected_hashtags = list(hashtags)
    if rng.random() > 0.4:
        bonus_niche = rng.choice(_NICHES)
        selected_hashtags.append(rng.choice(bonus_niche[1]))

    return {
        "video_id": f"v{index:04d}",
        "video_url": f"https://www.tiktok.com/@{username}/video/{700000000 + index}",
        "caption": _rand_caption(rng, niche_name),
        "hashtags": selected_hashtags,
        "keywords": keywords + [niche_name],
        "search_query": f"{niche_name} {rng.choice(keywords)}",
        "posted_at": posted_at.isoformat(),
        "likes": likes,
        "comments_count": num_comments,
        "shares": shares,
        "views": views,
        "author": {
            "author_id": f"a{1000 + index}",
            "username": username,
            "followers": rng.randint(1_000, 3_000_000),
        },
        "audio": {
            "audio_id": f"au{2000 + index}",
            "audio_title": rng.choice(_AUDIO_TITLES),
        },
        "video_meta": {
            "duration_seconds": rng.choice([15, 20, 25, 30, 45, 60, 90, 120, 180, 300]),
            "language": rng.choice(_LANGUAGES),
        },
        "comments": comments,
    }


def generate_dataset(
    count: int = 50,
    seed: int = 42,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
) -> List[Dict[str, Any]]:
    """Generate *count* synthetic TikTokPost dicts, fully reproducible via *seed*."""
    rng = random.Random(seed)
    if time_start is None:
        time_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if time_end is None:
        time_end = datetime(2026, 3, 1, tzinfo=timezone.utc)

    posts: List[Dict[str, Any]] = []
    for i in range(1, count + 1):
        post = generate_post(rng, index=i, time_range=(time_start, time_end))
        TikTokPost.model_validate(post)
        posts.append(post)
    return posts


def write_jsonl(posts: List[Dict[str, Any]], output_path: Path) -> None:
    """Write posts to a JSONL file, one JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for post in posts:
            fh.write(json.dumps(post, ensure_ascii=False) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TikTok JSONL data for testing.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=50,
        help="Number of records to generate (default: 50).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL path (default: data/mock/tiktok_posts_mock.jsonl).",
    )
    args = parser.parse_args()

    from src.common.constants import MOCK_DATA_PATH
    output_path = Path(args.output) if args.output else MOCK_DATA_PATH

    posts = generate_dataset(count=args.count, seed=args.seed)
    write_jsonl(posts, output_path)

    print(f"Generated {len(posts)} records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
