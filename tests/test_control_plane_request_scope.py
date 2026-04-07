from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_drift_monitor import _collect_request_id_scope as _collect_drift_scope
from scripts.run_experiment_analysis import _collect_request_id_scope as _collect_experiment_scope
from scripts.run_outcome_attribution import _collect_request_id_scope as _collect_outcome_scope
from scripts.run_retrain_controller import _collect_request_id_scope as _collect_retrain_scope


REQUEST_ID_A = "019d3f88-1bfb-7be5-8568-93da88e1b3c5"
REQUEST_ID_B = "019d3f88-2bd7-736a-99a0-00d98ebbd69a"


def _write_ids_payload(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_collect_request_scope_merges_cli_and_json_for_outcome(tmp_path):
    ids_path = _write_ids_payload(
        tmp_path / "request_ids.json",
        {"request_ids": [REQUEST_ID_A, REQUEST_ID_B]},
    )
    scoped = _collect_outcome_scope([REQUEST_ID_A], ids_path)
    assert scoped == [REQUEST_ID_A, REQUEST_ID_B]


def test_collect_request_scope_accepts_list_json_for_drift(tmp_path):
    ids_path = _write_ids_payload(tmp_path / "request_ids.json", [REQUEST_ID_A, REQUEST_ID_B])
    scoped = _collect_drift_scope([], ids_path)
    assert scoped == [REQUEST_ID_A, REQUEST_ID_B]


def test_collect_request_scope_rejects_invalid_uuid_for_experiment(tmp_path):
    ids_path = _write_ids_payload(tmp_path / "request_ids.json", {"request_ids": ["not-a-uuid"]})
    with pytest.raises(ValueError):
        _collect_experiment_scope([], ids_path)


def test_collect_request_scope_merges_cli_and_json_for_retrain(tmp_path):
    ids_path = _write_ids_payload(tmp_path / "request_ids.json", [REQUEST_ID_B])
    scoped = _collect_retrain_scope([REQUEST_ID_A], ids_path)
    assert scoped == [REQUEST_ID_A, REQUEST_ID_B]
