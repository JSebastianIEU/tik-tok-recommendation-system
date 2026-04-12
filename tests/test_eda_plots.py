from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from eda.src.plots import save_plot_template  # noqa: E402


def test_save_plot_template_png(tmp_path: Path):
    out = tmp_path / "fig.png"
    p = save_plot_template(output_path=out, title="T", metadata={"run_id": "r1"})
    assert p == out
    assert out.exists() and out.stat().st_size > 100
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_plot_template_md_embeds_png(tmp_path: Path):
    md = tmp_path / "report.md"
    save_plot_template(output_path=md, title="Report", metadata={"k": 1})
    png = tmp_path / "report.png"
    assert png.exists() and png.stat().st_size > 100
    text = md.read_text(encoding="utf-8")
    assert "# Report" in text
    assert "![Report](report.png)" in text
    assert "`k`" in text


def test_save_plot_template_bar_chart(tmp_path: Path):
    out = tmp_path / "bar.png"
    save_plot_template(
        output_path=out,
        title="Counts",
        metadata={
            "chart": "bar",
            "labels": ["a", "b"],
            "values": [3, 7],
            "y_label": "n",
        },
    )
    assert out.exists() and out.stat().st_size > 100
