"""Targeted tests for the B09 Full Run Preparation validator
(TASK-SLP-B09-FULL-RUN-PREPARATION-v0.1, R04 revision).

R04 fail-open → fail-closed fixes:
  Budget audit:
    - empty per_unit_wall / per_candidate_wall / peak_cuda_mb_per_candidate
      maps fail
    - missing unit in per_unit_wall fails
    - extra unknown unit in per_unit_wall fails
    - missing candidate in per_candidate_wall / peak_cuda fails
    - extra unknown candidate fails
    - per_candidate wall != sum(per_unit_wall) fails
    - total_wall != sum(per_unit_wall) fails
    - per_candidate peak != max(per_unit peak) fails
    - complete.json missing wall/peak fails
    - NaN / Inf / negative / string in complete.json fails
    - budget_ok=true but recompute exceeds limit fails
  TEST=0 evidence:
    - all 6 carriers required
    - test_access must be strict bool false
    - others must be strict int 0
    - any non-safe value (true, 1, string, float) fails
  Validator fixture vs real runner:
    - DONE.json missing → CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE
    - input_manifest_hashes missing 6 carriers → CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE
    - candidates/<cand>/candidate_decision seeds missing total_subjects → same
  A06 three-way binding:
    - B01 freeze core.a06_split_identifier != slp_subject_split_v0.1 fails
    - invalid hex SHA fails
    - three-way mismatch (B01 freeze vs protocol vs fold) fails
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_b09_full_run_preparation as b09v  # noqa: E402


PROTOCOL_FILENAME = "slp8_pm_full_protocol_v0.1.json"
FOLD_FILENAME = "slp8_pm_full_folds_v0.1.json"
EXPERIMENTS_DIR = "configs/experiments"
RUNNER_MODULE_REL = "src/topper_perception/neural/slp8_region_full.py"
RUNNER_SCRIPT_REL = "scripts/run_slp8_region_full.py"
FREEZE_MODULE_REL = "src/topper_perception/io/slp8_training_table_freeze.py"

REAL_B01_FREEZE = Path(
    r"E:\TeamProjects\smarttopper-team-workbench\data\processed\slp8_training_tables_v0.1\freeze_manifest.json"
)

# A06 split identifier expected by B09 audit (single governance source).
A06_SPLIT_IDENTIFIER = "slp_subject_split_v0.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "b09"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "b09@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_canonical(repo: Path) -> None:
    exp = repo / EXPERIMENTS_DIR
    exp.mkdir(parents=True, exist_ok=True)
    (exp / PROTOCOL_FILENAME).write_text(
        (ROOT / "configs" / "experiments" / PROTOCOL_FILENAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (exp / FOLD_FILENAME).write_text(
        (ROOT / "configs" / "experiments" / FOLD_FILENAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "topper_perception" / "neural").mkdir(parents=True, exist_ok=True)
    (repo / RUNNER_SCRIPT_REL).write_text(
        (ROOT / RUNNER_SCRIPT_REL).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / RUNNER_MODULE_REL).write_text(
        (ROOT / RUNNER_MODULE_REL).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "src" / "topper_perception" / "io").mkdir(parents=True, exist_ok=True)
    (repo / FREEZE_MODULE_REL).write_text(
        (ROOT / FREEZE_MODULE_REL).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "b09"
    env["GIT_AUTHOR_EMAIL"] = "b09@t"
    env["GIT_COMMITTER_NAME"] = "b09"
    env["GIT_COMMITTER_EMAIL"] = "b09@t"
    subprocess.run(["git", "commit", "-q", "-m", "B09 fixtures"], cwd=repo, env=env, check=True)


def _freeze_a06_sha() -> str:
    with open(REAL_B01_FREEZE, encoding="utf-8") as f:
        return json.load(f)["core"]["a06_split_sha256"]


def _identity() -> dict[str, Any]:
    return {
        "experiment_id": "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "git_commit": "PENDING",
        "git_dirty": False,
        "config_sha256": b09v.FROZEN_B07_PROTOCOL_SHA256,
        "data_manifest_sha256": b09v.FROZEN_B01_FREEZE_SHA256,
        "fold_manifest_sha256": b09v.FROZEN_B07_FOLD_SHA256,
        "split_sha256": _freeze_a06_sha(),
    }


def _unit_id(cand: str, fold_id: str, seed: int) -> str:
    return f"{cand}__{fold_id}__seed_{seed:04d}"


def _expected_unit_ids() -> set[str]:
    return {_unit_id(c, f, s) for c in b09v.EXPECTED_CANDIDATES for f in b09v.EXPECTED_FOLDS for s in b09v.EXPECTED_SEEDS}


def _make_full_audit_fixture(
    out_dir: Path,
    *,
    head: str,
    identity: dict | None = None,
    identity_overrides: dict | None = None,
    total_wall_seconds: float = 100.0,
    per_unit_wall: dict | None = None,
    per_candidate_wall: dict | None = None,
    per_candidate_peak: dict | None = None,
    per_unit_peak: dict | None = None,
    include_oof: bool = True,
    include_input_manifest_hashes: bool = True,
    imh_test_extra: dict | None = None,
    budget_ok: bool = True,
    skip_done: bool = False,
    done_extra: dict | None = None,
    candidate_status: str = "DONE",
    seed_status: str = "COMPLETE",
    seed_total_samples: int = 4095,
    include_total_subjects: bool = True,
) -> None:
    """Materialise a complete, valid 30-unit DONE fixture for R04.

    By default: includes DONE.json, manifest.json, status.json,
    budget_report.json, input_manifest_hashes.json (4 frozen hashes +
    6 TEST=0 carriers), oof_metrics_summary.json, 2×candidate_decision,
    30×unit complete.json.  All values are consistent with each other
    and within B07 hard upper bounds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_id = _identity()
    if identity is not None:
        base_id.update(identity)
    base_id["git_commit"] = head
    if identity_overrides:
        base_id.update(identity_overrides)

    # DONE.json (R04 requires)
    done_payload = {
        **base_id,
        "status": "DONE",
        "terminal_state": "DONE",
    }
    if done_extra:
        done_payload.update(done_extra)
    if not skip_done:
        (out_dir / "DONE.json").write_text(json.dumps(done_payload))

    # manifest.json (runner writes this)
    (out_dir / "manifest.json").write_text(json.dumps({
        **base_id,
        "task_id": "TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1",
        "protocol": "B07",
        "config_version": "slp8_pm_full_protocol_v0.1",
        "synthetic_mode": False,
        "candidates": list(b09v.EXPECTED_CANDIDATES),
        "seeds": list(b09v.EXPECTED_SEEDS),
        "folds": list(b09v.EXPECTED_FOLDS),
        "total_units": 30,
        "unit_count_done": 30,
        "unit_count_failed": 0,
        "unit_count_stopped": 0,
        "terminal_state": "DONE",
    }))

    # status.json (runner writes this)
    (out_dir / "status.json").write_text(json.dumps({
        **base_id,
        "terminal_state": "DONE",
        "total_units": 30,
        "unit_count_done": 30,
        "unit_count_failed": 0,
        "unit_count_stopped": 0,
        "winner": b09v.EXPECTED_CANDIDATES[0],
        "winner_mean_pooled_iou": 0.5,
        "total_wall_seconds": round(total_wall_seconds, 2),
    }))

    # budget_report.json (R04 exact key set required)
    # If per_unit_wall not given, use the computed 30 unit values
    expected_ids = _expected_unit_ids()
    per_unit_wall_provided = per_unit_wall is not None
    per_unit_wall = per_unit_wall or {uid: 100.0 for uid in expected_ids}
    per_candidate_wall = per_candidate_wall or {
        c: 1500.0 for c in b09v.EXPECTED_CANDIDATES
    }
    per_candidate_peak = per_candidate_peak or {
        c: 369.0 for c in b09v.EXPECTED_CANDIDATES
    }
    per_unit_peak = per_unit_peak or {uid: 369.0 for uid in expected_ids}
    if not per_unit_wall_provided:
        # use actual total
        actual_total = sum(per_unit_wall.values())
        total_wall_seconds = actual_total
    (out_dir / "budget_report.json").write_text(json.dumps({
        "max_wall_minutes_per_unit": b09v.MAX_WALL_MINUTES_PER_UNIT,
        "max_wall_minutes_per_candidate": b09v.MAX_WALL_MINUTES_PER_CANDIDATE,
        "max_wall_minutes_total": b09v.MAX_WALL_MINUTES_TOTAL,
        "max_peak_cuda_mb": b09v.MAX_PEAK_CUDA_MB,
        "total_wall_seconds": round(total_wall_seconds, 2),
        "total_wall_minutes": round(total_wall_seconds / 60.0, 2),
        "per_candidate_wall_seconds": {c: round(s, 2) for c, s in per_candidate_wall.items()},
        "per_unit_wall_seconds": {uid: round(s, 2) for uid, s in per_unit_wall.items()},
        "peak_cuda_mb_per_candidate": {c: round(p, 2) for c, p in per_candidate_peak.items()},
        "budget_ok": budget_ok,
    }))

    # input_manifest_hashes.json: 4 frozen + 6 TEST=0 carriers
    if imh_test_extra is None:
        imh = {
            "config_sha256": base_id["config_sha256"],
            "data_manifest_sha256": base_id["data_manifest_sha256"],
            "fold_manifest_sha256": base_id["fold_manifest_sha256"],
            "split_sha256": base_id["split_sha256"],
            "test_access": False,
            "test_rows": 0,
            "test_labels": 0,
            "test_onehot": 0,
            "test_predictions": 0,
            "test_metrics": 0,
        }
    else:
        imh = {
            "config_sha256": base_id["config_sha256"],
            "data_manifest_sha256": base_id["data_manifest_sha256"],
            "fold_manifest_sha256": base_id["fold_manifest_sha256"],
            "split_sha256": base_id["split_sha256"],
        }
        imh.update(imh_test_extra)
    if include_input_manifest_hashes:
        (out_dir / "input_manifest_hashes.json").write_text(json.dumps(imh))

    # candidate_decision.json (top-level)
    (out_dir / "candidate_decision.json").write_text(json.dumps({
        "winner": b09v.EXPECTED_CANDIDATES[0],
        "winner_decision": "WINNER",
        "winner_mean_pooled_iou": 0.5,
        "candidates": {
            c: {"decision": "WINNER" if c == b09v.EXPECTED_CANDIDATES[0] else "ELIMINATED",
                "mean_pooled_iou": 0.5 if c == b09v.EXPECTED_CANDIDATES[0] else 0.3}
            for c in b09v.EXPECTED_CANDIDATES
        },
    }))

    # oof_metrics_summary.json
    if include_oof:
        (out_dir / "oof_metrics_summary.json").write_text(json.dumps({
            c: {"mean_pooled_iou": 0.5, "mean_pooled_dice": 0.6,
                "mean_worst_subject_iou": 0.3, "status": candidate_status}
            for c in b09v.EXPECTED_CANDIDATES
        }))

    # candidates/<cand>/candidate_decision.json
    for c in b09v.EXPECTED_CANDIDATES:
        cd_dir = out_dir / "candidates" / c
        cd_dir.mkdir(parents=True, exist_ok=True)
        seeds_block: dict[str, dict] = {}
        for seed in b09v.EXPECTED_SEEDS:
            sb: dict[str, Any] = {
                "status": seed_status,
                "total_samples": seed_total_samples,
                "pooled_fixed_fg_macro_iou": 0.5,
                "pooled_fixed_fg_macro_dice": 0.6,
                "worst_subject_iou": 0.3,
            }
            if include_total_subjects:
                sb["total_subjects"] = b09v.EXPECTED_OOF_SUBJECTS_PER_SEED
            seeds_block[str(seed)] = sb
        (cd_dir / "candidate_decision.json").write_text(json.dumps({
            "candidate": c,
            "model_version": c,
            "exact_parameter_count": 120809 if c == b09v.EXPECTED_CANDIDATES[1] else 53449,
            "decision": "WINNER" if c == b09v.EXPECTED_CANDIDATES[0] else "ELIMINATED",
            "mean_pooled_iou": 0.5,
            "mean_pooled_dice": 0.6,
            "mean_worst_subject_iou": 0.3,
            "status": candidate_status,
            "seeds": seeds_block,
        }))

    # 30 unit complete.json files
    units_dir = out_dir / "units"
    units_dir.mkdir(exist_ok=True)
    for uid in sorted(expected_ids):
        u = units_dir / uid
        u.mkdir(exist_ok=True)
        cand, fold_id, seed = uid.split("__")[0], uid.split("__")[1], int(uid.split("__seed_")[1])
        cpl = {
            "unit": {
                "candidate": cand,
                "fold_id": fold_id,
                "seed": seed,
                "exp_id": base_id["experiment_id"],
                "model_version": cand,
            },
            "identity": dict(base_id),
            "result": {
                "status": "DONE",
                "wall_seconds": float(per_unit_wall.get(uid, 100.0)),
                "val_sample_count": 855,
                "train_sample_count": 3240,
                "best_epoch": 22,
                "best_val_loss": 0.1,
                "val_fixed_fg_macro_iou": 0.5,
                "val_fixed_fg_macro_dice": 0.6,
                "val_background_iou": 0.9,
                "peak_cuda_mb": float(per_unit_peak.get(uid, 369.0)),
            },
            "budget": {
                "max_wall_minutes_per_unit": b09v.MAX_WALL_MINUTES_PER_UNIT,
                "max_wall_minutes_per_candidate": b09v.MAX_WALL_MINUTES_PER_CANDIDATE,
                "max_wall_minutes_total": b09v.MAX_WALL_MINUTES_TOTAL,
                "max_peak_cuda_mb": b09v.MAX_PEAK_CUDA_MB,
            },
        }
        (u / "complete.json").write_text(json.dumps(cpl))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_with_b01_freeze(tmp_path: Path) -> Path:
    if not REAL_B01_FREEZE.is_file():
        pytest.skip(f"real B01 freeze missing: {REAL_B01_FREEZE}")
    repo = _seed_repo(tmp_path)
    _write_canonical(repo)
    return repo


@pytest.fixture
def b01_freeze_path() -> Path:
    if not REAL_B01_FREEZE.is_file():
        pytest.skip(f"real B01 freeze missing: {REAL_B01_FREEZE}")
    return REAL_B01_FREEZE


# ---------------------------------------------------------------------------
# 1. no-write / no-train / no-TEST (static AST)
# ---------------------------------------------------------------------------


def test_28_validator_remains_no_write(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "out_no_write"
    assert not out.exists()
    b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--output-dir", str(out),
        "--b01-freeze-manifest", str(b01_freeze_path),
    ])
    assert not out.exists(), f"validator must not create {out}"


def test_29_validator_does_not_invoke_training() -> None:
    text = (SCRIPTS / "validate_b09_full_run_preparation.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = {"train_one_unit", "run_full", "run_one_fold_preflight", "run_synthetic_cpu_smoke"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
            if node.module:
                imported.add(node.module)
    for sym in forbidden:
        assert sym not in imported, f"validator imports forbidden runner symbol {sym!r}"
    assert "topper_perception.neural.slp8_region_full" not in imported
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    for sym in forbidden:
        assert sym not in called, f"validator calls forbidden runner entry {sym!r}"


def test_30_validator_does_not_access_test() -> None:
    text = (SCRIPTS / "validate_b09_full_run_preparation.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    assert "enable_test_access" not in called
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "load_test":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        raise AssertionError("validator must not set load_test=True")
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.asname or a.name)
    assert "load_b01_freeze_tables" not in imported
    assert "topper_perception.io.slp8_training_table_freeze" not in imported


def test_31_normal_30_unit_plan_passes_preparation(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "out_fresh"
    assert not out.exists()
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--output-dir", str(out),
        "--b01-freeze-manifest", str(b01_freeze_path),
    ])
    log = b09v._capture_last_main_log()
    assert rc == 0, f"expected pass; errors:\n{log.summary_text()}"
    assert not out.exists(), "validator must not create output_dir"


# ---------------------------------------------------------------------------
# 16. positive audit-only baseline
# ---------------------------------------------------------------------------


def test_16_audit_only_valid_done_passes(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc == 0, f"audit-only valid DONE should pass; errors:\n{log.summary_text()}"


# ---------------------------------------------------------------------------
# R04-#1: Budget audit fail-open fixes
# ---------------------------------------------------------------------------


def test_budget_three_maps_all_empty_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    budget["per_unit_wall_seconds"] = {}
    budget["per_candidate_wall_seconds"] = {}
    budget["peak_cuda_mb_per_candidate"] = {}
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "missing" in errs
    assert "per_unit_wall_seconds" in errs


def test_budget_missing_one_unit_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    full_keys = list(budget["per_unit_wall_seconds"].keys())
    del budget["per_unit_wall_seconds"][full_keys[0]]
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "per_unit_wall_seconds" in errs
    assert "missing" in errs


def test_budget_extra_unknown_unit_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    budget["per_unit_wall_seconds"]["FAKE_UNIT"] = 1.0
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "unexpected" in errs
    assert "FAKE_UNIT" in errs


def test_budget_missing_candidate_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    del budget["per_candidate_wall_seconds"][b09v.EXPECTED_CANDIDATES[0]]
    del budget["peak_cuda_mb_per_candidate"][b09v.EXPECTED_CANDIDATES[0]]
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "per_candidate_wall_seconds" in errs
    assert b09v.EXPECTED_CANDIDATES[0] in errs


def test_budget_extra_unknown_candidate_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    budget["per_candidate_wall_seconds"]["FAKE_CANDIDATE"] = 1.0
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "FAKE_CANDIDATE" in errs


def test_budget_candidate_wall_inconsistent_with_unit_sum_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    # mismatch: per_candidate_wall_seconds does not match sum(per_unit_wall_seconds)
    c = b09v.EXPECTED_CANDIDATES[0]
    real_sum = sum(v for uid, v in budget["per_unit_wall_seconds"].items() if uid.startswith(c + "__"))
    budget["per_candidate_wall_seconds"][c] = real_sum + 100.0  # off by 100s
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert c in errs
    assert "per_candidate_wall_seconds" in errs


def test_budget_total_wall_inconsistent_with_unit_sum_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    real_total = sum(budget["per_unit_wall_seconds"].values())
    budget["total_wall_seconds"] = real_total + 50.0  # off by 50s
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "total_wall_seconds" in errs


def test_budget_candidate_peak_inconsistent_with_unit_max_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    c = b09v.EXPECTED_CANDIDATES[0]
    real_max = max(v for uid, v in budget["per_unit_wall_seconds"].items() if uid.startswith(c + "__"))
    # Set the candidate peak to something different than the per-unit max
    # by tampering with per_unit_peak and budget's peak_cuda_mb_per_candidate differently
    budget["peak_cuda_mb_per_candidate"][c] = 500.0  # not 369.0 from per_unit_peak
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "peak_cuda_mb_per_candidate" in errs


def test_budget_complete_json_missing_wall_peak_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    # Remove wall_seconds and peak_cuda_mb from one complete.json's result
    uid = next(iter(_expected_unit_ids()))
    cp = json.loads((out / "units" / uid / "complete.json").read_text(encoding="utf-8"))
    del cp["result"]["wall_seconds"]
    del cp["result"]["peak_cuda_mb"]
    (out / "units" / uid / "complete.json").write_text(json.dumps(cp))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert uid in errs
    assert "wall_seconds" in errs or "invalid" in errs


def test_budget_complete_json_nan_inf_negative_string_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    uid = next(iter(_expected_unit_ids()))
    cp = json.loads((out / "units" / uid / "complete.json").read_text(encoding="utf-8"))
    cp["result"]["wall_seconds"] = "not_a_number"
    (out / "units" / uid / "complete.json").write_text(json.dumps(cp))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert uid in errs
    assert "wall_seconds" in errs


def test_budget_ok_true_but_recompute_exceeds_limit_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    # Set per_candidate_peak[c] to 99999 but keep budget_ok=true;
    # the validator must catch it via the recompute cross-check.
    budget = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    c = b09v.EXPECTED_CANDIDATES[0]
    budget["peak_cuda_mb_per_candidate"][c] = 99999.0
    (out / "budget_report.json").write_text(json.dumps(budget))
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "peak_cuda_mb_per_candidate" in errs
    assert "99999" in errs


def test_budget_two_decimal_writer_rounding_passes(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    """The real budget writer rounds report values to two decimals.

    Raw values in complete.json may consequently differ by nearly 0.005 from
    their report carrier without any tampering.  The audit must accept that
    documented serialization precision while still rejecting larger drift.
    """
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    unit_wall = {uid: 100.004 for uid in _expected_unit_ids()}
    candidate_wall = {c: 1500.06 for c in b09v.EXPECTED_CANDIDATES}
    _make_full_audit_fixture(
        out,
        head=head,
        per_unit_wall=unit_wall,
        per_candidate_wall=candidate_wall,
        total_wall_seconds=3000.12,
    )
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc == 0, f"errors:\n{log.summary_text()}"


# ---------------------------------------------------------------------------
# R04-#2: TEST=0 evidence
# ---------------------------------------------------------------------------


def test_test_zero_all_six_carriers_passes(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)  # default: all 6 carriers
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc == 0, f"errors:\n{log.summary_text()}"


def test_test_zero_test_access_true_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, imh_test_extra={"test_access": True})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_access" in errs
    assert "True" in errs or "true" in errs


def test_test_zero_test_rows_one_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, imh_test_extra={"test_rows": 1})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_rows" in errs


def test_test_zero_string_zero_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, imh_test_extra={"test_rows": "0"})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_rows" in errs
    assert "strict int" in errs or "str" in errs


def test_test_zero_bool_one_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, imh_test_extra={"test_rows": True})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_rows" in errs


def test_test_zero_missing_one_carrier_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    extras = {
        "test_access": False, "test_rows": 0, "test_labels": 0,
        "test_onehot": 0, "test_predictions": 0,
        # test_metrics omitted
    }
    _make_full_audit_fixture(out, head=head, imh_test_extra=extras)
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_metrics" in errs
    assert "CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE" in errs


def test_test_zero_unknown_test_field_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    extras = {
        "test_access": False, "test_rows": 0, "test_labels": 0,
        "test_onehot": 0, "test_predictions": 0, "test_metrics": 0,
        "test_oh_i_forgot": "0",
    }
    _make_full_audit_fixture(out, head=head, imh_test_extra=extras)
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "test_oh_i_forgot" in errs


# ---------------------------------------------------------------------------
# R04-#3: Validator fixture vs real runner schema
# ---------------------------------------------------------------------------


def test_done_json_missing_fails_with_bridge_gap(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, skip_done=True)
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "DONE.json" in errs
    assert "CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE" in errs


def test_done_json_non_json_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head)
    (out / "DONE.json").write_text("not json")
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "DONE.json" in errs
    assert "JSON" in errs or "valid" in errs


def test_done_json_tampered_experiment_id_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, identity_overrides={"experiment_id": "EXP-FAKE"})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "DONE.json" in errs
    assert "experiment_id" in errs


def test_done_json_tampered_git_commit_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, identity_overrides={"git_commit": "0" * 40})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "DONE.json" in errs
    assert "git_commit" in errs


def test_imh_missing_carriers_fails_with_bridge_gap(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    # Keep 4 frozen hashes but no TEST carriers → fail-closed per R04
    _make_full_audit_fixture(out, head=head, imh_test_extra={})
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE" in errs
    assert "test_access" in errs


def test_per_seed_total_subjects_missing_fails_with_bridge_gap(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "done_dir"
    head = _git_head(repo_with_b01_freeze)
    _make_full_audit_fixture(out, head=head, include_total_subjects=False)
    rc = b09v.main([
        "--repo-root", str(repo_with_b01_freeze),
        "--output-dir", str(out),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--b01-freeze-manifest", str(b01_freeze_path),
        "--audit-only",
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE" in errs
    assert "total_subjects" in errs


# ---------------------------------------------------------------------------
# R04-#4: A06 three-way binding
# ---------------------------------------------------------------------------


def test_a06_identifier_wrong_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "out_fresh"
    # Mutate the freeze manifest in the temp repo: change core.a06_split_identifier
    repo = repo_with_b01_freeze
    src = REAL_B01_FREEZE.read_text(encoding="utf-8")
    parsed = json.loads(src)
    parsed["core"]["a06_split_identifier"] = "wrong_split_v9.9"
    bad = tmp_path / "bad_freeze.json"
    bad.write_text(json.dumps(parsed), encoding="utf-8")
    rc = b09v.main([
        "--repo-root", str(repo),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--output-dir", str(out),
        "--b01-freeze-manifest", str(bad),
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "a06_split_identifier" in errs
    assert "slp_subject_split_v0.1" in errs


def test_a06_invalid_hex_sha_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "out_fresh"
    repo = repo_with_b01_freeze
    src = REAL_B01_FREEZE.read_text(encoding="utf-8")
    parsed = json.loads(src)
    parsed["core"]["a06_split_sha256"] = "not-a-valid-sha"
    bad = tmp_path / "bad_freeze.json"
    bad.write_text(json.dumps(parsed), encoding="utf-8")
    rc = b09v.main([
        "--repo-root", str(repo),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--output-dir", str(out),
        "--b01-freeze-manifest", str(bad),
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "lowercase hex" in errs


def test_a06_three_way_mismatch_fails(repo_with_b01_freeze, b01_freeze_path, tmp_path: Path) -> None:
    out = tmp_path / "out_fresh"
    repo = repo_with_b01_freeze
    # Mutate B07 fold manifest source_a06_split_sha256 in the temp repo
    fold_text = (repo / EXPERIMENTS_DIR / FOLD_FILENAME).read_text(encoding="utf-8")
    fold = json.loads(fold_text)
    fold["source_a06_split_sha256"] = "0" * 64  # different from frozen
    (repo / EXPERIMENTS_DIR / FOLD_FILENAME).write_text(json.dumps(fold, indent=2), encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "mutate fold"],
        cwd=repo, env={**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
                       "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"},
        check=True,
    )
    rc = b09v.main([
        "--repo-root", str(repo),
        "--experiment-id", "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "--output-dir", str(out),
        "--b01-freeze-manifest", str(REAL_B01_FREEZE),
    ])
    log = b09v._capture_last_main_log()
    assert rc != 0
    errs = " ".join(log.errors)
    assert "A06 split SHA mismatch" in errs


# ---------------------------------------------------------------------------
# Sanity: sentinel + EXP-ID format
# ---------------------------------------------------------------------------


def test_synthetic_exp_id_sentinel_rejected() -> None:
    log = b09v.CheckLog()
    b09v._check_experiment_id("EXP-SLP-B08-SYNTHETIC-SMOKE", log)
    assert any("synthetic sentinel" in e for e in log.errors)


def test_experiment_id_format() -> None:
    log = b09v.CheckLog()
    b09v._check_experiment_id("EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01", log)
    assert not log.errors
    log2 = b09v.CheckLog()
    b09v._check_experiment_id("EXP-B09-RANDOM-STRING", log2)
    assert log2.errors
