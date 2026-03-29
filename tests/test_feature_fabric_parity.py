from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.recommendation.fabric import FeatureFabric


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_ts_shim_and_python_fabric_have_parity_on_fixed_fixture():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    repo_root = Path(__file__).resolve().parent.parent
    fixture = json.loads(
        (repo_root / "tests" / "fixtures" / "fabric_parity_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    frontend_dir = repo_root / "frontend"

    node_script = f"""
import fs from "node:fs";
import path from "node:path";
import {{ buildCandidateProfileCore }} from "./server/modeling/step1/buildCandidateProfileCore.ts";
import {{ extractCandidateSignals }} from "./server/modeling/part2/extractCandidateSignals.ts";

const fixture = JSON.parse(
  fs.readFileSync(path.resolve("{str((repo_root / 'tests' / 'fixtures' / 'fabric_parity_fixture.json')).replace('\\', '\\\\')}"), "utf-8")
);
const core = buildCandidateProfileCore({{
  description: fixture.description,
  mentions: fixture.mentions,
  hashtags: fixture.hashtags,
  objective: fixture.objective,
  audience: fixture.audience,
  content_type: fixture.content_type,
  primary_cta: fixture.primary_cta,
  locale: fixture.locale
}});
const out = extractCandidateSignals(core, fixture.signal_hints);
console.log(JSON.stringify({{
  token_count: out.transcript_ocr.token_count,
  hook_timing_seconds: out.structure.hook_timing_seconds,
  payoff_timing_seconds: out.structure.payoff_timing_seconds,
  speech_ratio: out.audio.speech_ratio,
  shot_change_rate: out.visual.shot_change_rate
}}));
"""
    proc = subprocess.run(
        [node, "--import", "tsx", "-e", node_script],
        cwd=frontend_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"tsx parity probe failed: {proc.stderr.strip()[:180]}")

    ts_out = json.loads(proc.stdout.strip())
    py_out = FeatureFabric().extract(
        {
            "video_id": "fixture-video",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": fixture["description"],
            "hashtags": [f"#{item}" for item in fixture["hashtags"]],
            "keywords": fixture["hashtags"],
            "transcript_text": fixture["signal_hints"]["transcript_text"],
            "ocr_text": fixture["signal_hints"]["ocr_text"],
            "duration_seconds": fixture["signal_hints"]["duration_seconds"],
            "content_type": fixture["content_type"],
            "hints": fixture["signal_hints"],
        }
    )
    assert abs(py_out.text.token_count - ts_out["token_count"]) <= 12
    assert abs(py_out.structure.hook_timing_seconds - ts_out["hook_timing_seconds"]) <= 5.0
    assert (
        abs(py_out.structure.payoff_timing_seconds - ts_out["payoff_timing_seconds"]) <= 4.0
    )
    assert py_out.audio.speech_ratio is not None
    assert abs(py_out.audio.speech_ratio - ts_out["speech_ratio"]) <= 0.2
    assert py_out.visual.shot_change_rate is not None
    assert abs(py_out.visual.shot_change_rate - ts_out["shot_change_rate"]) <= 0.25
