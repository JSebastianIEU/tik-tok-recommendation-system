from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DatasetName = Literal["full", "videos", "comments", "authors"]

CONTRACTS: dict[DatasetName, set[str]] = {
    "videos": {"video_id", "created_at", "video_author_id"},
    "comments": {"comment_id", "video_id", "text", "row_created_at"},
    "authors": {"author_id", "row_created_at"},
    "full": {"video_id", "created_at", "comments"},
}

SILVER_CONTRACTS: dict[DatasetName, set[str]] = {
    "videos": {"video_id", "created_at", "video_author_id", "engagement_rate"},
    "comments": {"comment_id", "video_id", "text", "row_created_at", "text_length", "is_reply"},
    "authors": {"author_id", "row_created_at", "videos_count", "comments_count", "comments_per_video"},
    "full": {"video_id", "created_at", "comments", "comment_count_extracted", "engagement_rate"},
}


@dataclass(frozen=True)
class ContractValidationResult:
    dataset: DatasetName
    rows_checked: int
    missing_required_columns: list[str]
    row_missing_required_count: int
    valid: bool


def validate_dataset_rows(dataset: DatasetName, rows: list[dict[str, Any]]) -> ContractValidationResult:
    required = CONTRACTS[dataset]
    if not rows:
        return ContractValidationResult(
            dataset=dataset,
            rows_checked=0,
            missing_required_columns=[],
            row_missing_required_count=0,
            valid=True,
        )

    first = rows[0]
    missing_cols = sorted(col for col in required if col not in first)

    row_missing = 0
    for row in rows:
        if any(row.get(col) is None for col in required):
            row_missing += 1

    return ContractValidationResult(
        dataset=dataset,
        rows_checked=len(rows),
        missing_required_columns=missing_cols,
        row_missing_required_count=row_missing,
        valid=(not missing_cols and row_missing == 0),
    )


def validate_silver_rows(dataset: DatasetName, rows: list[dict[str, Any]]) -> ContractValidationResult:
    required = SILVER_CONTRACTS[dataset]
    if not rows:
        return ContractValidationResult(
            dataset=dataset,
            rows_checked=0,
            missing_required_columns=[],
            row_missing_required_count=0,
            valid=True,
        )

    first = rows[0]
    missing_cols = sorted(col for col in required if col not in first)

    row_missing = 0
    for row in rows:
        if any(col not in row for col in required):
            row_missing += 1

    return ContractValidationResult(
        dataset=dataset,
        rows_checked=len(rows),
        missing_required_columns=missing_cols,
        row_missing_required_count=row_missing,
        valid=(not missing_cols and row_missing == 0),
    )
