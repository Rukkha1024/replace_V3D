"""Tests for inverse-dynamics config selection parsing."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scripts.run_batch_all_timeseries_csv import (
    _load_force_assignment_config,
    _load_inverse_dynamics_forceplate_selection,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8-sig")


def test_forceplate_selection_rejects_duplicates(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    _write_yaml(
        cfg,
        {
            "forceplate": {
                "analysis": {
                    "use_for_inverse_dynamics": ["fp1", "fp1"],
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Duplicate inverse-dynamics forceplate selection"):
        _load_inverse_dynamics_forceplate_selection(cfg)


def test_force_assignment_config_loads_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    _write_yaml(
        cfg,
        {
            "forceplate": {
                "analysis": {
                    "use_for_inverse_dynamics": ["fp1", "fp3"],
                    "force_assignment": {
                        "cop_distance_threshold_m": 0.25,
                        "remove_incomplete_assignments": False,
                        "require_segment_projection_on_plate": True,
                        "log_assignment_summary": False,
                    },
                    "inertia": {"model": "per_segment_rog_v1"},
                }
            }
        },
    )

    conf = _load_force_assignment_config(cfg)
    assert conf.cop_distance_threshold_m == 0.25
    assert conf.remove_incomplete_assignments is False
    assert conf.require_segment_projection_on_plate is True
    assert conf.log_assignment_summary is False
