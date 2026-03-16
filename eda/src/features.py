from __future__ import annotations

from typing import Any


def build_feature_templates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Template-only feature transform entrypoint.

    This is intentionally lightweight scaffolding (not a concrete feature pipeline).
    Add dataset-specific transforms here when EDA hypotheses are stable.
    """
    return rows
