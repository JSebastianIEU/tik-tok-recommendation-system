from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class ArtifactRegistry:
    root_dir: Path

    def _manifest_path(self, bundle_dir: Path) -> Path:
        return bundle_dir / "manifest.json"

    def create_bundle_dir(self, run_name: Optional[str] = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = f"{stamp}-{(run_name or 'recommender').strip().replace(' ', '-')}"
        bundle_dir = self.root_dir / folder
        bundle_dir.mkdir(parents=True, exist_ok=True)
        return bundle_dir

    def write_manifest(
        self,
        bundle_dir: Path,
        payload: Dict[str, Any],
    ) -> None:
        manifest_path = self._manifest_path(bundle_dir)
        enriched = {
            **payload,
            "created_at": _utc_now_iso(),
        }
        manifest_path.write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    def load_manifest(self, bundle_dir: Path) -> Dict[str, Any]:
        return json.loads(self._manifest_path(bundle_dir).read_text(encoding="utf-8"))

    def feature_schema_hash(self, feature_names: list[str]) -> str:
        canonical = json.dumps(sorted(feature_names), separators=(",", ":"), ensure_ascii=False)
        return _sha256_text(canonical)

    def assert_compatible(
        self,
        bundle_dir: Path,
        expected: Dict[str, Any],
    ) -> Dict[str, Any]:
        manifest = self.load_manifest(bundle_dir)
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = manifest.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"{key}: expected={expected_value!r}, actual={actual_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Artifact compatibility check failed: " + " | ".join(mismatches)
            )
        return manifest
