from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scraper.data_requests import DATASET_LOADERS, RetrievalPage, _fetch_all_pages, export_rows

DatasetName = Literal["full", "videos", "comments", "authors"]


@dataclass(frozen=True)
class LoadRequest:
    dataset: DatasetName
    limit: int = 1000
    since: str | None = None
    all_pages: bool = False
    fmt: Literal["jsonl", "csv"] = "jsonl"


def fetch_dataset(*, request: LoadRequest, db_url: str) -> tuple[list[dict], dict | None]:
    if request.all_pages:
        rows = _fetch_all_pages(
            dataset=request.dataset,
            db_url=db_url,
            page_limit=request.limit,
            since=request.since,
        )
        return rows, None

    page: RetrievalPage = DATASET_LOADERS[request.dataset](
        db_url=db_url,
        limit=request.limit,
        offset=0,
        since=request.since,
        cursor_created_at=None,
        cursor_id=None,
    )
    return page.rows, page.next_cursor


def write_extract(*, rows: list[dict], path: Path, fmt: Literal["jsonl", "csv"]) -> None:
    export_rows(rows, output_path=path, fmt=fmt)
