"""Shared path constants used across scripts and modules.

All paths are absolute ``pathlib.Path`` objects derived from the repository
root so callers don't need to assemble them manually.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# -- data -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
MOCK_DATA_DIR = DATA_DIR / "mock"
MOCK_DATA_PATH = MOCK_DATA_DIR / "tiktok_posts_mock.jsonl"

# -- artifacts (all gitignored) -------------------------------------------
ARTIFACTS_DIR = ROOT / "artifacts"
CONTRACTS_DIR = ARTIFACTS_DIR / "contracts"
DATAMART_DIR = ARTIFACTS_DIR / "datamart"
RECOMMENDER_DIR = ARTIFACTS_DIR / "recommender"
RECOMMENDER_LATEST = RECOMMENDER_DIR / "latest"
COMMENT_INTELLIGENCE_DIR = ARTIFACTS_DIR / "comment_intelligence"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
LABELING_SESSIONS_DIR = ARTIFACTS_DIR / "labeling_sessions"
CONTROL_PLANE_DIR = ARTIFACTS_DIR / "control_plane"
PIPELINE_REPORT_PATH = ARTIFACTS_DIR / "pipeline_report.json"

# -- reports / EDA --------------------------------------------------------
REPORTS_DIR = ROOT / "src" / "baseline"
DEFAULT_REPORT_PATH = REPORTS_DIR / "report.md"
EDA_DIR = ROOT / "eda"
EDA_EXTRACTS_DIR = EDA_DIR / "extracts"
EDA_REPORTS_DIR = EDA_DIR / "reports"
