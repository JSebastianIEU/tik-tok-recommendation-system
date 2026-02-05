import json
from pathlib import Path
from typing import Tuple, List

from .schemas import TikTokPost

def validate_file(jsonl_path: Path) -> Tuple[int, List[str]]:
    """Validate mocked JSONL records against TikTokPost schema."""
    errors: List[str] = []
    count = 0
    for line_no, raw in enumerate(jsonl_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            TikTokPost.model_validate_json(raw)
            count += 1
        except Exception as exc:  # TODO: tighten exception types
            errors.append(f"Line {line_no}: {exc}")
    return count, errors

# TODO: add CLI-friendly reporter with severity levels and data quality scores.
