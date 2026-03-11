from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_run_registry(manifest: dict[str, Any], *, registry_path: Path) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, default=str))
        f.write("\n")
