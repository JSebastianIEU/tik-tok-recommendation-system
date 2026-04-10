from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_human_comparable_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_human_comparable_benchmark", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_row_hashtags_falls_back_to_caption_extraction():
    module = _load_builder_module()
    row = {
        "caption": "Mini tutorial fácil 🙌🏼💫🎀 #tutorial #beauty #maquillaje",
        "hashtags": [],
    }
    assert module._row_hashtags(row) == ["#tutorial", "#beauty", "#maquillaje"]
