from pathlib import Path
import pandas as pd

def compute_stats(jsonl_path: Path) -> dict:
    """Compute lightweight descriptive stats on mocked data."""
    df = pd.read_json(jsonl_path, lines=True)
    # TODO: expand with engagement metrics, language breakdowns, hashtag frequency.
    return {
        "records": len(df),
        "avg_likes": float(df["likes"].mean()),
        "avg_views": float(df["views"].mean()),
        "avg_comments": float(df["comments_count"].mean()),
    }

def write_report(stats: dict, report_path: Path) -> None:
    lines = [
        "# Baseline Report (Mock)\n",
        "## Summary\n",
        f"- Records: {stats['records']}",
        f"- Avg likes: {stats['avg_likes']:.1f}",
        f"- Avg views: {stats['avg_views']:.1f}",
        f"- Avg comments: {stats['avg_comments']:.1f}",
        "\n## TODO\n- Add trend plots and author-level aggregates.\n",
    ]
    report_path.write_text("\n".join(lines))

# TODO: plug into richer EDA notebooks or dashboards later.
