from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _figure_ext(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".png", ".svg", ".pdf"}:
        return ext
    return ".png"


def _build_figure(title: str, metadata: dict[str, Any] | None) -> plt.Figure:
    meta = metadata or {}
    fig, ax = plt.subplots(figsize=(8, 5))

    if meta.get("chart") == "bar":
        labels = meta.get("labels") or []
        values = meta.get("values") or []
        if (
            isinstance(labels, list)
            and isinstance(values, list)
            and len(labels) == len(values)
            and len(values) > 0
        ):
            ax.bar(range(len(labels)), values, color="#2ecc71", edgecolor="#27ae60")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([str(x) for x in labels], rotation=45, ha="right")
            y_label = meta.get("y_label", "Value")
            ax.set_ylabel(str(y_label))
            ax.set_title(title, fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            return fig

    ax.axis("off")
    lines = [title, ""]
    for k, v in sorted(meta.items(), key=lambda kv: str(kv[0])):
        if k in ("chart", "labels", "values"):
            continue
        lines.append(f"{k}: {v}")
    if len(lines) <= 2:
        lines.append("(no extra metadata)")
    text = "\n".join(lines)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    fig.tight_layout()
    return fig


def save_plot_template(*, output_path: Path, title: str, metadata: dict[str, Any] | None = None) -> Path:
    """Render a Matplotlib figure and save it, optionally wrapped in Markdown.

    When ``metadata`` contains ``chart: "bar"`` plus matching ``labels`` and
    ``values`` lists, a bar chart is drawn. Otherwise a title + metadata text
    panel is drawn.

    If ``output_path`` ends in ``.md``, a raster image is written next to it
    (default ``.png``) and the markdown embeds it. For ``.png`` / ``.svg`` /
    ``.pdf``, the figure is saved to that path. Other extensions are replaced
    with ``.png``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata) if metadata else {}

    fig = _build_figure(title, meta)
    try:
        if output_path.suffix.lower() == ".md":
            img_path = output_path.with_suffix(_figure_ext(output_path))
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            rel = img_path.name
            lines = [f"# {title}", "", f"![{title}]({rel})", ""]
            if meta:
                lines.append("## Metadata")
                for k, v in sorted(meta.items(), key=lambda kv: str(kv[0])):
                    lines.append(f"- `{k}`: {v}")
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return output_path

        target = output_path
        if target.suffix.lower() not in {".png", ".svg", ".pdf"}:
            target = output_path.with_suffix(".png")
        fig.savefig(target, dpi=150, bbox_inches="tight")
        return target
    finally:
        plt.close(fig)
