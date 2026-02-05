import sys
from pathlib import Path

from src.baseline.baseline_stats import compute_stats, write_report
from src.common.constants import ROOT

def main() -> int:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "mock" / "tiktok_posts_mock.jsonl"
    report_path = ROOT / "src" / "baseline" / "report.md"
    stats = compute_stats(jsonl_path)
    write_report(stats, report_path)
    print(f"Wrote baseline report to {report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TODO: parameterize report output path and add CLI flags for more metrics.
