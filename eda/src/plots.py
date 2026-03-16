from __future__ import annotations

from pathlib import Path
from typing import Any


def save_plot_template(*, output_path: Path, title: str, metadata: dict[str, Any] | None = None) -> Path:
    """Template-only plot artifact writer.

    This writes a markdown placeholder so teams have a stable output contract
    before chart logic is finalized.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "Plot template placeholder."]
    if metadata:
        lines.append("")
        lines.append("## Metadata")
        for k, v in metadata.items():
            lines.append(f"- {k}: {v}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
