from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def execute_notebook(notebook_path: Path, *, output_path: Path, parameters: dict[str, str] | None = None) -> None:
    """Execute notebook headlessly if papermill is available.

    This keeps notebook execution pipeline-ready while allowing local fallback.
    """
    if shutil.which("papermill") is None:
        raise RuntimeError("papermill is not installed. Install EDA extras to execute notebooks.")

    cmd = ["papermill", str(notebook_path), str(output_path)]
    for key, value in (parameters or {}).items():
        cmd.extend(["-p", key, value])
    subprocess.run(cmd, check=True)
