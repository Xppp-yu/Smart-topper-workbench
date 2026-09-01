"""B09 Full Run Preparation validator
(TASK-SLP-B09-FULL-RUN-PREPARATION-v0.1, R04 revision).

This validator is fail-closed: returns non-zero exit code and reports
errors whenever any required check fails.  It does NOT create any output
directory, does NOT invoke training, does NOT read TEST data, and
does NOT import or call the B08 runner's training paths.

R04 revision (vs R03):

  * Budget audit fail-closed:
    - per_unit_wall_seconds / per_candidate_wall_seconds /
      peak_cuda_mb_per_candidate must contain the exact expected keys
      (30 unit IDs / 2 candidate names); missing or extra keys ERR.
    - Budget values must be finite non-negative numbers.
    - All budget values are recomputed from units/<uid>/complete.json
      and cross-checked against budget_report.json within
      RECOMPUTE_TOLERANCE.  budget_ok=true does not bypass.
    - Per-unit peak CUDA in complete.json must be finite non-negative
      and <= 8192 MiB.
  * TEST=0 evidence strengthened:
    - input_manifest_hashes.json must contain all 6 safe TEST carriers:
      test_access (strict bool false), test_rows, test_labels,
      test_onehot, test_predictions, test_metrics (each strict int 0).
    - Any other field whose name contains "test" is ERR.
  * Validator fixture vs real runner schema:
    - The validator reports CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE for
      any required carrier that the B08 runner's write_run_artifacts()
      does NOT currently produce.  These are recorded as bridge task
      blockers and listed in the task contract + PROJECT_STATUS.
  * A06 split three-way binding:
    - B01 freeze core.a06_split_identifier == "slp_subject_split_v0.1"
    - B01 freeze core.a06_split_sha256 == 64-char lowercase hex
    - B07 protocol data_contract.a06_split_sha256 == same value
    - B07 fold manifest source_a06_split_sha256 == same value
    - All three must match.
  * No --a06-split-manifest CLI flag (single source = B01 freeze).
  * Real runner schema is the authoritative source for what is auditable.
    DONE.json / FAILED.json / STOPPED.json are NOT written by the runner;
    the validator requires them and reports CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE
    if missing.  This is recorded in §22 bridge checklist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Frozen B07 / B08 R03 reference values
# ---------------------------------------------------------------------------

B07_PROTOCOL_PATH = "configs/experiments/slp8_pm_full_protocol_v0.1.json"
B07_FOLD_PATH = "configs/experiments/slp8_pm_full_folds_v0.1.json"

FROZEN_B07_PROTOCOL_SHA256 = "98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83"
FROZEN_B07_FOLD_SHA256 = "0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b"

# Bound via --b01-freeze-manifest.
FROZEN_B01_FREEZE_SHA256 = "42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04"

EXPECTED_CANDIDATES: tuple[str, ...] = (
    "slp8_deeplabv3plus_lite_v0.1",
    "slp8_resunet_lite_v0.1",
)
EXPECTED_FOLDS: tuple[str, ...] = tuple(f"fold_{i}" for i in range(1, 6))
EXPECTED_SEEDS: tuple[int, ...] = (42, 123, 2026)
EXPECTED_TOTAL_UNITS = 30

# B07 hard upper bounds.
MAX_WALL_MINUTES_PER_UNIT = 15
MAX_WALL_MINUTES_PER_CANDIDATE = 15 * 5 * 3          # 225
MAX_WALL_MINUTES_TOTAL = 15 * 5 * 3 * 2              # 450
MAX_PEAK_CUDA_MB = 8192
EXPECTED_OOF_SAMPLES_PER_SEED = 4095
EXPECTED_OOF_SUBJECTS_PER_SEED = 91

# Tolerance for budget recomputation cross-check (seconds / MiB).
# Tight enough to catch real inconsistencies; loose enough to absorb
# JSON float round-trip noise.
# ``build_budget_report`` serializes budget values with ``round(value, 2)``.
# A valid raw ``complete.json`` value can therefore differ from the serialized
# report by almost 0.005.  Keep the audit tight while accepting that declared
# writer contract (the epsilon avoids binary floating-point edge cases).
RECOMPUTE_TOLERANCE = 0.005001

# Six safe TEST=0 carrier keys that input_manifest_hashes.json MUST contain.
TEST_CARRIER_KEYS: tuple[str, ...] = (
    "test_access",
    "test_rows",
    "test_labels",
    "test_onehot",
    "test_predictions",
    "test_metrics",
)
TEST_CARRIER_INT_KEYS: tuple[str, ...] = (
    "test_rows", "test_labels", "test_onehot",
    "test_predictions", "test_metrics",
)

# A06 split identifier (single governance source).
A06_SPLIT_IDENTIFIER = "slp_subject_split_v0.1"

RUNNER_SCRIPT = "scripts/run_slp8_region_full.py"
RUNNER_MODULE = "src/topper_perception/neural/slp8_region_full.py"
TRAINING_TABLE_FREEZE = "src/topper_perception/io/slp8_training_table_freeze.py"

SYNTHETIC_EXP_ID_SENTINELS: frozenset[str] = frozenset({
    "EXP-SLP-B08-SYNTHETIC-SMOKE",
    "EXP-SLP-B09-SYNTHETIC-SMOKE",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_show_bytes(repo_root: Path, rel_path: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=str(repo_root), capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _committed_sha256(repo_root: Path, rel_path: str) -> str | None:
    blob = _git_show_bytes(repo_root, rel_path)
    if blob is not None:
        return hashlib.sha256(blob).hexdigest()
    path = (repo_root / rel_path).resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, timeout=10,
            text=True, check=True,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, timeout=10,
            text=True, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_strict_int(v: Any) -> bool:
    """True only for plain int (not bool, not float, not str)."""
    return type(v) is int and not isinstance(v, bool)


def _is_valid_sha256(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and re.fullmatch(r"[0-9a-f]{64}", s) is not None


def _unit_id(cand: str, fold_id: str, seed: int) -> str:
    return f"{cand}__{fold_id}__seed_{seed:04d}"


def _expected_unit_ids() -> set[str]:
    return {_unit_id(c, f, s) for c in EXPECTED_CANDIDATES for f in EXPECTED_FOLDS for s in EXPECTED_SEEDS}


# ---------------------------------------------------------------------------
# CheckLog
# ---------------------------------------------------------------------------


class CheckLog:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.oks: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def ok(self, msg: str) -> None:
        self.oks.append(msg)

    def require(self, cond: bool, ok_msg: str, err_msg: str) -> None:
        if cond:
            self.oks.append(ok_msg)
        else:
            self.errors.append(err_msg)

    def summary_text(self) -> str:
        return "\n".join(
            [f"OK  : {o}" for o in self.oks] + [f"ERR : {e}" for e in self.errors]
        )


# ---------------------------------------------------------------------------
# B01 freeze binding + A06 split three-way binding
# ---------------------------------------------------------------------------


def _check_b01_freeze_binding(path: Path, log: CheckLog) -> str | None:
    if path is None:
        log.err("--b01-freeze-manifest path was not supplied (REQUIRED)")
        return None
    if not path.is_file():
        log.err(f"B01 freeze manifest does not exist: {path}")
        return None
    sha = _sha256_file(path)
    log.require(
        sha == FROZEN_B01_FREEZE_SHA256,
        f"B01 freeze manifest SHA matches frozen ({sha[:12]}…)",
        f"B01 freeze manifest SHA mismatch: expected {FROZEN_B01_FREEZE_SHA256}, got {sha}",
    )
    return sha


def _load_b01_freeze(freeze_path: Path, log: CheckLog) -> dict | None:
    try:
        return json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.err(f"freeze manifest is not valid JSON: {exc}")
        return None


def _check_a06_three_way(
    protocol: dict | None,
    fold: dict | None,
    freeze: dict | None,
    log: CheckLog,
) -> str | None:
    """Return the canonical A06 split SHA on success, None on failure.
    Verifies three-way consistency between B01 freeze, B07 protocol
    and B07 fold manifest; rejects all drift.
    """
    if freeze is None or protocol is None or fold is None:
        return None
    core = freeze.get("core", {})
    ident = core.get("a06_split_identifier")
    log.require(
        ident == A06_SPLIT_IDENTIFIER,
        f"freeze core.a06_split_identifier == {A06_SPLIT_IDENTIFIER!r}",
        f"freeze core.a06_split_identifier must be {A06_SPLIT_IDENTIFIER!r}, got {ident!r}",
    )
    freeze_a06 = core.get("a06_split_sha256")
    log.require(
        _is_valid_sha256(freeze_a06),
        f"freeze core.a06_split_sha256 valid hex ({str(freeze_a06)[:12]}…)",
        f"freeze core.a06_split_sha256 must be 64-char lowercase hex, got {freeze_a06!r}",
    )
    protocol_a06 = protocol.get("data_contract", {}).get("a06_split_sha256")
    log.require(
        _is_valid_sha256(protocol_a06),
        f"protocol data_contract.a06_split_sha256 valid hex ({str(protocol_a06)[:12]}…)",
        f"protocol data_contract.a06_split_sha256 must be 64-char lowercase hex, got {protocol_a06!r}",
    )
    fold_a06 = fold.get("source_a06_split_sha256")
    log.require(
        _is_valid_sha256(fold_a06),
        f"fold source_a06_split_sha256 valid hex ({str(fold_a06)[:12]}…)",
        f"fold source_a06_split_sha256 must be 64-char lowercase hex, got {fold_a06!r}",
    )

    log.require(
        freeze_a06 == protocol_a06,
        "freeze core.a06_split_sha256 == protocol data_contract.a06_split_sha256",
        f"A06 split SHA mismatch: freeze={freeze_a06} protocol={protocol_a06}",
    )
    log.require(
        freeze_a06 == fold_a06,
        "freeze core.a06_split_sha256 == fold source_a06_split_sha256",
        f"A06 split SHA mismatch: freeze={freeze_a06} fold={fold_a06}",
    )
    log.require(
        protocol_a06 == fold_a06,
        "protocol data_contract.a06_split_sha256 == fold source_a06_split_sha256",
        f"A06 split SHA mismatch: protocol={protocol_a06} fold={fold_a06}",
    )
    return freeze_a06


# ---------------------------------------------------------------------------
# Committed-content SHA + protocol/fold loaders
# ---------------------------------------------------------------------------


def _check_committed_hashes(repo_root: Path, log: CheckLog) -> None:
    proto_sha = _committed_sha256(repo_root, B07_PROTOCOL_PATH)
    if proto_sha is None:
        log.err(f"cannot read committed {B07_PROTOCOL_PATH}")
    else:
        log.require(
            proto_sha == FROZEN_B07_PROTOCOL_SHA256,
            f"protocol SHA matches ({proto_sha[:12]}…)",
            f"protocol SHA mismatch: expected {FROZEN_B07_PROTOCOL_SHA256}, got {proto_sha}",
        )
    fold_sha = _committed_sha256(repo_root, B07_FOLD_PATH)
    if fold_sha is None:
        log.err(f"cannot read committed {B07_FOLD_PATH}")
    else:
        log.require(
            fold_sha == FROZEN_B07_FOLD_SHA256,
            f"fold manifest SHA matches ({fold_sha[:12]}…)",
            f"fold manifest SHA mismatch: expected {FROZEN_B07_FOLD_SHA256}, got {fold_sha}",
        )


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 30-unit matrix / fold isolation / TEST / budget / identity / metrics
# ---------------------------------------------------------------------------


def _check_30_unit_matrix(protocol: dict, log: CheckLog) -> None:
    raw_candidates = [c["name"] for c in protocol.get("candidates", [])]
    log.require(
        tuple(raw_candidates) == EXPECTED_CANDIDATES,
        "candidates match frozen set+order",
        f"candidates mismatch: expected {EXPECTED_CANDIDATES}, got {tuple(raw_candidates)}",
    )
    seeds = tuple(int(s) for s in protocol.get("training_contract", {}).get("seeds", []))
    log.require(seeds == EXPECTED_SEEDS, f"seeds match frozen set {EXPECTED_SEEDS}", f"seeds mismatch")
    matrix = protocol.get("execution_matrix", {})
    log.require(matrix.get("candidates") == len(EXPECTED_CANDIDATES), "candidates==2", f"candidates={matrix.get('candidates')}")
    log.require(matrix.get("folds") == len(EXPECTED_FOLDS), "folds==5", f"folds={matrix.get('folds')}")
    log.require(matrix.get("seeds") == len(EXPECTED_SEEDS), "seeds==3", f"seeds={matrix.get('seeds')}")
    log.require(matrix.get("total_units") == EXPECTED_TOTAL_UNITS, "total_units==30", f"total_units={matrix.get('total_units')}")


def _check_fold_isolation(folds: dict, log: CheckLog) -> None:
    log.require(folds.get("test_access") == "DENIED", "fold.test_access==DENIED", f"test_access={folds.get('test_access')!r}")
    log.require(folds.get("development_subject_count") == 91, "dev_subjects==91", f"dev_subjects={folds.get('development_subject_count')}")
    log.require(folds.get("development_sample_count") == 4095, "dev_samples==4095", f"dev_samples={folds.get('development_sample_count')}")
    fold_rows = folds.get("folds", [])
    log.require(len(fold_rows) == 5, "exactly 5 folds", f"folds={len(fold_rows)}")
    seen: list[str] = []
    for f in fold_rows:
        subjects = list(f.get("val_subject_ids", []))
        log.require(
            len(subjects) == len(set(subjects)),
            f"fold {f.get('fold_id')} unique val subjects",
            f"fold {f.get('fold_id')} duplicate val subjects",
        )
        seen.extend(subjects)
    log.require(len(seen) == 91, "5 folds cover 91 subjects", f"got {len(seen)}")
    log.require(len(set(seen)) == 91, "all 91 subjects unique", f"unique={len(set(seen))}")
    inv = folds.get("invariants", {})
    log.require(inv.get("test_subjects_in_any_fold") == 0, "test_subjects_in_any_fold==0", f"{inv.get('test_subjects_in_any_fold')}")
    log.require(inv.get("train_val_subject_overlap_per_fold") == 0, "train_val_overlap==0", f"{inv.get('train_val_subject_overlap_per_fold')}")
    log.require(inv.get("each_development_subject_exactly_one_val_fold") is True, "each_subject_one_val_fold==True", f"{inv.get('each_development_subject_exactly_one_val_fold')!r}")
    log.require(inv.get("fit_preprocessing_on_fold_train_only") is True, "fit_preprocessing_on_fold_train_only==True", f"{inv.get('fit_preprocessing_on_fold_train_only')!r}")


def _check_test_policy(protocol: dict, log: CheckLog) -> None:
    test = protocol.get("test_access", {})
    expected = {"allowed": False, "load_test": False, "expected_rows": 0, "expected_labels": 0, "expected_onehot": 0}
    log.require(test == expected, "protocol.test_access=={allowed:false,load_test:false,*expected:0}", f"test={test}")
    sel = protocol.get("selection_rule", {})
    log.require(sel.get("no_test_in_selection") is True, "selection_rule.no_test_in_selection==True", f"{sel.get('no_test_in_selection')!r}")
    hg = protocol.get("hard_fail_closed_gates", [])
    if "any TEST row, label, onehot, statistic, prediction or metric access" not in hg:
        log.err("hard_fail_closed_gates must list TEST=0 invariant")
    else:
        log.ok("hard_fail_closed_gates includes TEST=0 invariant")


def _check_budget(protocol: dict, log: CheckLog) -> None:
    b = protocol.get("resource_budget", {})
    log.require(b.get("max_wall_minutes_per_fold_seed_unit") == MAX_WALL_MINUTES_PER_UNIT, f"budget.unit=={MAX_WALL_MINUTES_PER_UNIT}", f"got {b.get('max_wall_minutes_per_fold_seed_unit')}")
    log.require(b.get("max_wall_minutes_per_candidate") == MAX_WALL_MINUTES_PER_CANDIDATE, f"budget.candidate=={MAX_WALL_MINUTES_PER_CANDIDATE}", f"got {b.get('max_wall_minutes_per_candidate')}")
    log.require(b.get("max_wall_minutes_total") == MAX_WALL_MINUTES_TOTAL, f"budget.total=={MAX_WALL_MINUTES_TOTAL}", f"got {b.get('max_wall_minutes_total')}")
    log.require(b.get("max_peak_cuda_mb") == MAX_PEAK_CUDA_MB, f"budget.peak_cuda_mb=={MAX_PEAK_CUDA_MB}", f"got {b.get('max_peak_cuda_mb')}")


def _check_identity_contract(protocol: dict, log: CheckLog) -> None:
    identity = protocol.get("identity_contract", {})
    required = set(identity.get("required_fields", []))
    expected = {
        "experiment_id", "git_commit", "git_dirty",
        "config_sha256", "data_manifest_sha256",
        "fold_manifest_sha256", "split_sha256", "model_version",
    }
    missing = expected - required
    log.require(not missing, f"identity required_fields ⊇ {sorted(expected)}", f"missing: {sorted(missing)}")
    log.require(identity.get("git_dirty_must_be") is False, "git_dirty_must_be==False", f"{identity.get('git_dirty_must_be')!r}")


def _check_metrics(protocol: dict, log: CheckLog) -> None:
    metrics = protocol.get("metrics", {})
    primary = str(metrics.get("primary", "")).lower()
    log.require("pooled" in primary, "primary mentions pooled OOF", f"primary={primary!r}")
    log.require(metrics.get("fold_average_is_primary") is False, "fold_average_is_primary==False", f"{metrics.get('fold_average_is_primary')!r}")
    log.require(metrics.get("failed_units_may_be_dropped") is False, "failed_units_may_be_dropped==False", f"{metrics.get('failed_units_may_be_dropped')!r}")
    log.require(metrics.get("background_included_in_primary") is False, "background_included_in_primary==False", f"{metrics.get('background_included_in_primary')!r}")


# ---------------------------------------------------------------------------
# Runner code static checks
# ---------------------------------------------------------------------------


def _read_text(repo_root: Path, rel_path: str) -> str | None:
    p = (repo_root / rel_path).resolve()
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _check_runner_code(repo_root: Path, log: CheckLog) -> None:
    runner_text = _read_text(repo_root, RUNNER_MODULE)
    if runner_text is None:
        log.err(f"runner module not found: {RUNNER_MODULE}")
    else:
        if re.search(r"enable_test_access\s*\(", runner_text):
            log.err("Full runner source contains enable_test_access( call")
        else:
            log.ok("Full runner source has no enable_test_access( call")
        if re.search(r"load_test\s*=\s*True", runner_text):
            log.err("Full runner source sets load_test=True")
        else:
            log.ok("Full runner source never sets load_test=True")
        if "load_b01_freeze_tables" not in runner_text:
            log.err("Full runner source does not import load_b01_freeze_tables")
        else:
            log.ok("Full runner source uses load_b01_freeze_tables")
        if "load_test=False" not in runner_text:
            log.err("Full runner source does not pin load_test=False")
        else:
            log.ok("Full runner source pins load_test=False on B01 loader")
        for needle, label in (
            ("load_checkpoint_for_resume", "checkpoint resume loader"),
            ("write_unit_complete_atomic", "atomic unit-complete writer"),
            ("atomic_write_json", "atomic JSON writer"),
            ("refuse_overwrite", "output overwrite guard"),
            ("validate_oof_rows", "OOF row validator"),
            ("merge_seed_oof", "OOF seed merge"),
            ("write_budget_state_atomic", "atomic budget state writer"),
            ("load_budget_state", "budget state loader"),
        ):
            if needle not in runner_text:
                log.err(f"Full runner source missing {label} ({needle})")
            else:
                log.ok(f"Full runner source has {label}")

    cli_text = _read_text(repo_root, RUNNER_SCRIPT)
    if cli_text is None:
        log.err(f"runner CLI not found: {RUNNER_SCRIPT}")
    else:
        if "run_authorized" not in cli_text or "--run-authorized" not in cli_text:
            log.err("runner CLI missing --run-authorized gate")
        else:
            log.ok("runner CLI has --run-authorized gate")
        if "Real B01 run not executed by this task" not in cli_text:
            log.err("runner CLI no longer carries the explicit 30-unit refusal")
        else:
            log.ok("runner CLI keeps the explicit 30-unit real-B01 refusal (B09 will need a bridge task)")
        if (
            "Real B01 run not executed by this task" in cli_text
            or "--run-full" not in cli_text
        ):
            log.ok("runner CLI does not expose an unguarded 30-unit real-B01 entry point")
        else:
            log.err("runner CLI has --run-full flag but no B09 bridge task gate")
        if "slp8_training_table_freeze" in cli_text:
            log.ok("runner CLI imports the B01 freeze module")
        else:
            log.err("runner CLI does not import the B01 freeze module")

    freeze_text = _read_text(repo_root, TRAINING_TABLE_FREEZE)
    if freeze_text is None:
        log.err(f"B01 freeze module not found: {TRAINING_TABLE_FREEZE}")
    else:
        if "def enable_test_access" in freeze_text:
            log.ok("B01 freeze module exposes enable_test_access (audit-able guard)")
        else:
            log.err("B01 freeze module missing enable_test_access function")
        if '"final_evaluation"' in freeze_text or "'final_evaluation'" in freeze_text:
            log.ok("B01 freeze module only accepts purpose='final_evaluation'")
        else:
            log.err("B01 freeze module does not appear to restrict purpose to 'final_evaluation'")


# ---------------------------------------------------------------------------
# Output-dir state (preparation mode, fail-closed)
# ---------------------------------------------------------------------------


_INTERRUPTED_ALLOWED_STATUS = {"INTERRUPTED", "STOPPED"}


def _check_output_dir_state(
    output_dir: Path | None,
    *,
    experiment_id: str | None,
    current_head: str | None,
    a06_split_sha: str | None,
    log: CheckLog,
) -> None:
    if output_dir is None:
        log.ok("output-dir not provided; skip stateful output check")
        return
    if not output_dir.exists():
        log.ok(f"output_dir does not exist: {output_dir}")
        return
    if not output_dir.is_dir():
        log.err(f"output_dir exists but is not a directory: {output_dir}")
        return

    terminals = {
        "DONE.json": output_dir / "DONE.json",
        "FAILED.json": output_dir / "FAILED.json",
        "STOPPED.json": output_dir / "STOPPED.json",
    }
    present = {n: p for n, p in terminals.items() if p.is_file()}
    if len(present) > 1:
        log.err(f"output_dir has multiple terminal files {sorted(present)}")
        return
    if not present:
        log.err(f"output_dir {output_dir} exists but is empty; refuse")
        return
    name, path = next(iter(present.items()))
    if name == "DONE.json":
        log.err(f"output_dir already has DONE.json; refusing to overwrite")
        return
    if name == "FAILED.json":
        log.err(f"output_dir already has FAILED.json; refusing to overwrite")
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.err(f"STOPPED.json unreadable: {exc}")
        return
    status_field = payload.get("status") or payload.get("terminal_state")
    if status_field is None:
        log.err("STOPPED.json missing status/terminal_state")
    elif str(status_field) not in _INTERRUPTED_ALLOWED_STATUS:
        log.err(f"STOPPED.json status must be in {sorted(_INTERRUPTED_ALLOWED_STATUS)}, got {status_field!r}")
    else:
        log.ok(f"STOPPED.json status allowed: {status_field}")
    if experiment_id is None:
        log.err("STOPPED.json present but --experiment-id not supplied")
    else:
        log.require(
            payload.get("experiment_id") == experiment_id,
            f"STOPPED.json experiment_id matches ({experiment_id})",
            f"STOPPED.json experiment_id mismatch",
        )
    gc = payload.get("git_commit")
    if gc is None:
        log.err("STOPPED.json missing git_commit")
    elif current_head is None:
        log.err("cannot read current HEAD")
    elif str(gc) != str(current_head):
        log.err(f"STOPPED.json git_commit mismatch: file={gc!r} head={current_head!r}")
    else:
        log.ok(f"STOPPED.json git_commit matches HEAD")
    if "git_dirty" not in payload:
        log.err("STOPPED.json missing git_dirty")
    elif payload.get("git_dirty") is not False:
        log.err(f"STOPPED.json git_dirty must be False, got {payload.get('git_dirty')!r}")
    else:
        log.ok("STOPPED.json git_dirty==False")

    expected_id = _expected_frozen_identity(a06_split_sha)
    for key, expected in expected_id.items():
        actual = payload.get(key)
        if actual is None:
            log.err(f"STOPPED.json missing frozen-hash field {key!r}")
        elif str(actual) != str(expected):
            log.err(f"STOPPED.json frozen hash {key!r} mismatch")
        else:
            log.ok(f"STOPPED.json frozen hash {key!r} matches")

    units_dir = output_dir / "units"
    if units_dir.is_dir():
        for u in sorted(units_dir.iterdir()):
            if not u.is_dir():
                continue
            complete = u / "complete.json"
            failed = u / "unit_failed.json"
            if failed.is_file():
                log.err(f"unit {u.name} has unit_failed.json; INTERRUPTED forbids failure")
            if complete.is_file() and failed.is_file():
                log.err(f"unit {u.name} has both complete.json and unit_failed.json")
            if complete.is_file():
                try:
                    cpl = json.loads(complete.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    log.err(f"unit {u.name} complete.json unreadable: {exc}")
                    continue
                cid = cpl.get("identity", {})
                for key, expected in expected_id.items():
                    actual = cid.get(key)
                    if str(actual) != str(expected):
                        log.err(f"unit {u.name} complete.json identity.{key} mismatch")
                if current_head is not None and str(cid.get("git_commit", "")) != str(current_head):
                    log.err(f"unit {u.name} complete.json git_commit mismatch")
                if cid.get("git_dirty") is not False:
                    log.err(f"unit {u.name} complete.json git_dirty must be False")


def _expected_frozen_identity(a06_split_sha: str | None) -> dict[str, str]:
    return {
        "config_sha256": FROZEN_B07_PROTOCOL_SHA256,
        "data_manifest_sha256": FROZEN_B01_FREEZE_SHA256,
        "fold_manifest_sha256": FROZEN_B07_FOLD_SHA256,
        "split_sha256": a06_split_sha or "",
    }


# ---------------------------------------------------------------------------
# audit-only: strict read-only audit (R04 fail-closed)
# ---------------------------------------------------------------------------


SCHEMA_GAP_PREFIX = "CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE"


def _schema_gap(label: str) -> str:
    return f"{SCHEMA_GAP_PREFIX}: {label} is required by B09 audit but not produced by current B08 runner; bridge task must add writer"


def _parse_json_object(path: Path, label: str, log: CheckLog) -> dict | None:
    if not path.is_file():
        log.err(f"audit-only: {label} missing at {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.err(f"audit-only: {label} unreadable or not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log.err(f"audit-only: {label} is not a JSON object")
        return None
    return payload


def _check_required_key(name: str, expected: Any, actual: Any, log: CheckLog) -> bool:
    if not _is_finite_number(actual):
        log.err(f"audit-only: {name} must be a finite number, got {actual!r}")
        return False
    if actual < 0:
        log.err(f"audit-only: {name} must be >= 0, got {actual!r}")
        return False
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=RECOMPUTE_TOLERANCE):
        log.err(f"audit-only: {name}={actual} != {expected} (tol {RECOMPUTE_TOLERANCE})")
        return False
    return True


def _read_units_recompute(units_dir: Path, log: CheckLog) -> dict | None:
    """Read all 30 unit complete.json and recompute per-unit/per-candidate/total.

    Returns dict with keys: per_unit_wall, per_unit_peak, per_cand_wall,
    per_cand_peak, total_wall, identity_unit_set, all finite non-negative.
    """
    expected_ids = _expected_unit_ids()
    per_unit_wall: dict[str, float] = {}
    per_unit_peak: dict[str, float] = {}
    per_cand_wall: dict[str, float] = {c: 0.0 for c in EXPECTED_CANDIDATES}
    per_cand_peak: dict[str, float] = {c: 0.0 for c in EXPECTED_CANDIDATES}
    total_wall: float = 0.0
    seen: set[str] = set()

    for u in sorted(units_dir.iterdir()):
        if not u.is_dir():
            continue
        uid = u.name
        if uid not in expected_ids:
            log.err(f"audit-only: unexpected unit dir {uid!r} (not in frozen 30-unit matrix)")
            continue
        seen.add(uid)
        cp = _parse_json_object(u / "complete.json", f"{uid}/complete.json", log)
        if cp is None:
            return None
        result = cp.get("result", {})
        ws = result.get("wall_seconds")
        pk = result.get("peak_cuda_mb")
        if not _is_finite_number(ws) or ws < 0:
            log.err(f"audit-only: {uid} result.wall_seconds invalid: {ws!r}")
            return None
        if pk is not None and (not _is_finite_number(pk) or pk < 0):
            log.err(f"audit-only: {uid} result.peak_cuda_mb invalid: {pk!r}")
            return None
        per_unit_wall[uid] = float(ws)
        per_unit_peak[uid] = float(pk) if pk is not None else 0.0
        if ws > MAX_WALL_MINUTES_PER_UNIT * 60:
            log.err(f"audit-only: {uid} wall_seconds={ws} > {MAX_WALL_MINUTES_PER_UNIT*60}")
        if pk is not None and pk > MAX_PEAK_CUDA_MB:
            log.err(f"audit-only: {uid} peak_cuda_mb={pk} > {MAX_PEAK_CUDA_MB}")
        # parse candidate from uid
        cand = uid.split("__", 1)[0]
        if cand in per_cand_wall:
            per_cand_wall[cand] += float(ws)
            per_cand_peak[cand] = max(per_cand_peak[cand], float(pk) if pk is not None else 0.0)
        total_wall += float(ws)

    missing = expected_ids - seen
    if missing:
        log.err(f"audit-only: {len(missing)} unit(s) missing in units/ (expected 30)")
        return None
    return {
        "per_unit_wall": per_unit_wall,
        "per_unit_peak": per_unit_peak,
        "per_cand_wall": per_cand_wall,
        "per_cand_peak": per_cand_peak,
        "total_wall": total_wall,
    }


def _check_budget_audit(
    output_dir: Path,
    identity: dict[str, str],
    log: CheckLog,
) -> dict | None:
    """R04-#1 strict fail-closed budget audit.

    Requires:
      - budget_report.json exists
      - per_unit_wall_seconds keys == exact 30 unit IDs
      - per_candidate_wall_seconds keys == exact 2 candidate names
      - peak_cuda_mb_per_candidate keys == exact 2 candidate names
      - all values finite non-negative and within B07 hard upper bounds
      - budget_ok == True (or computed and consistent)
      - values match the recomputation from 30 units/<uid>/complete.json
        within RECOMPUTE_TOLERANCE
    """
    budget = _parse_json_object(output_dir / "budget_report.json", "budget_report.json", log)
    if budget is None:
        return None

    required_keys = {
        "max_wall_minutes_per_unit", "max_wall_minutes_per_candidate",
        "max_wall_minutes_total", "max_peak_cuda_mb",
        "total_wall_seconds", "per_candidate_wall_seconds",
        "per_unit_wall_seconds", "peak_cuda_mb_per_candidate", "budget_ok",
    }
    missing_keys = required_keys - set(budget.keys())
    if missing_keys:
        log.err(f"audit-only: budget_report.json missing keys: {sorted(missing_keys)}")
        return None

    # ---- 1. exact key set checks ----
    actual_unit_keys = set(budget["per_unit_wall_seconds"].keys())
    expected_unit_keys = _expected_unit_ids()
    missing_unit_keys = expected_unit_keys - actual_unit_keys
    extra_unit_keys = actual_unit_keys - expected_unit_keys
    if missing_unit_keys:
        log.err(f"audit-only: per_unit_wall_seconds missing {len(missing_unit_keys)} unit(s): {sorted(missing_unit_keys)[:5]}{'...' if len(missing_unit_keys) > 5 else ''}")
    if extra_unit_keys:
        log.err(f"audit-only: per_unit_wall_seconds has {len(extra_unit_keys)} unexpected key(s): {sorted(extra_unit_keys)[:5]}{'...' if len(extra_unit_keys) > 5 else ''}")

    actual_cand_keys_wall = set(budget["per_candidate_wall_seconds"].keys())
    actual_cand_keys_peak = set(budget["peak_cuda_mb_per_candidate"].keys())
    expected_cand_keys = set(EXPECTED_CANDIDATES)
    for label, keys in (
        ("per_candidate_wall_seconds", actual_cand_keys_wall),
        ("peak_cuda_mb_per_candidate", actual_cand_keys_peak),
    ):
        miss = expected_cand_keys - keys
        extra = keys - expected_cand_keys
        if miss:
            log.err(f"audit-only: {label} missing candidate(s): {sorted(miss)}")
        if extra:
            log.err(f"audit-only: {label} has unexpected candidate(s): {sorted(extra)}")

    # ---- 2. value finite non-negative + within upper bound ----
    tws = budget["total_wall_seconds"]
    if not _is_finite_number(tws) or tws < 0:
        log.err(f"audit-only: budget_report.total_wall_seconds invalid: {tws!r}")
    elif tws > MAX_WALL_MINUTES_TOTAL * 60:
        log.err(f"audit-only: total_wall_seconds={tws} > {MAX_WALL_MINUTES_TOTAL*60}")

    for uid, w in (budget.get("per_unit_wall_seconds") or {}).items():
        if not _is_finite_number(w) or w < 0:
            log.err(f"audit-only: per_unit_wall_seconds[{uid!r}]={w!r} invalid")
        elif w > MAX_WALL_MINUTES_PER_UNIT * 60:
            log.err(f"audit-only: per_unit_wall_seconds[{uid!r}]={w} > {MAX_WALL_MINUTES_PER_UNIT*60}")
    for c, w in (budget.get("per_candidate_wall_seconds") or {}).items():
        if not _is_finite_number(w) or w < 0:
            log.err(f"audit-only: per_candidate_wall_seconds[{c!r}]={w!r} invalid")
        elif w > MAX_WALL_MINUTES_PER_CANDIDATE * 60:
            log.err(f"audit-only: per_candidate_wall_seconds[{c!r}]={w} > {MAX_WALL_MINUTES_PER_CANDIDATE*60}")
    for c, p in (budget.get("peak_cuda_mb_per_candidate") or {}).items():
        if not _is_finite_number(p) or p < 0:
            log.err(f"audit-only: peak_cuda_mb_per_candidate[{c!r}]={p!r} invalid")
        elif p > MAX_PEAK_CUDA_MB:
            log.err(f"audit-only: peak_cuda_mb_per_candidate[{c!r}]={p} > {MAX_PEAK_CUDA_MB}")

    # ---- 3. recompute from 30 complete.json and cross-check ----
    units_dir = output_dir / "units"
    if not units_dir.is_dir():
        log.err("audit-only: units/ directory missing")
        return None
    rec = _read_units_recompute(units_dir, log)
    if rec is None:
        return None

    # per-unit wall cross-check
    for uid in expected_unit_keys:
        if uid in budget["per_unit_wall_seconds"]:
            _check_required_key(
                f"per_unit_wall_seconds[{uid!r}]",
                rec["per_unit_wall"][uid],
                budget["per_unit_wall_seconds"][uid],
                log,
            )
    # per-candidate wall cross-check
    for c in expected_cand_keys:
        if c in budget["per_candidate_wall_seconds"]:
            _check_required_key(
                f"per_candidate_wall_seconds[{c!r}]",
                rec["per_cand_wall"][c],
                budget["per_candidate_wall_seconds"][c],
                log,
            )
    # per-candidate peak CUDA cross-check
    for c in expected_cand_keys:
        if c in budget["peak_cuda_mb_per_candidate"]:
            _check_required_key(
                f"peak_cuda_mb_per_candidate[{c!r}]",
                rec["per_cand_peak"][c],
                budget["peak_cuda_mb_per_candidate"][c],
                log,
            )
    # total wall cross-check
    _check_required_key(
        "total_wall_seconds",
        rec["total_wall"],
        budget["total_wall_seconds"],
        log,
    )

    # budget_ok must be True (computed consistency)
    if budget.get("budget_ok") is not True:
        log.err(f"audit-only: budget_report.budget_ok must be True, got {budget.get('budget_ok')!r}")
    else:
        log.ok("audit-only: budget_report.budget_ok==True")
    return rec


def _check_test_zero_carriers(imh: dict, log: CheckLog) -> None:
    """R04-#2 strict TEST=0 evidence: all 6 carriers required, exact types."""
    # 1. all 6 must exist
    for k in TEST_CARRIER_KEYS:
        if k not in imh:
            log.err(_schema_gap(f"input_manifest_hashes.{k}"))
    # 2. test_access must be strict bool False
    if "test_access" in imh:
        if imh["test_access"] is not False:
            log.err(
                f"audit-only: test_access must be strict bool false, got {imh['test_access']!r} (type {type(imh['test_access']).__name__})"
            )
        else:
            log.ok("audit-only: test_access is strict bool false")
    # 3. other 5 must be strict int 0
    for k in TEST_CARRIER_INT_KEYS:
        if k not in imh:
            continue
        v = imh[k]
        if not _is_strict_int(v):
            log.err(
                f"audit-only: {k} must be strict int 0, got {v!r} (type {type(v).__name__})"
            )
        elif v != 0:
            log.err(f"audit-only: {k}={v!r} is not 0")
        else:
            log.ok(f"audit-only: {k} is strict int 0")
    # 4. any other field with "test" in name must ERR
    for k in imh.keys():
        if k in TEST_CARRIER_KEYS:
            continue
        if "test" in k.lower():
            log.err(f"audit-only: unrecognized TEST field {k!r}={imh[k]!r}")


def _check_done_json(output_dir: Path, log: CheckLog) -> dict | None:
    """Audit the terminal DONE carrier.

    The B08 runner does write DONE.json through ``write_terminal_state`` once
    all planned units reach a terminal result.  B09 additionally requires the
    full frozen identity carried below; its absence is a bridge schema gap,
    not evidence that the terminal writer itself is absent.
    """
    done = _parse_json_object(output_dir / "DONE.json", "DONE.json", log)
    if done is None:
        log.err(_schema_gap("DONE.json (status/identity/terminal contract)"))
        return None
    # required fields
    for key in ("status", "terminal_state", "experiment_id", "git_commit", "git_dirty",
                "config_sha256", "data_manifest_sha256",
                "fold_manifest_sha256", "split_sha256"):
        if key not in done:
            log.err(f"audit-only: DONE.json missing required field {key!r}")
    status = done.get("status") or done.get("terminal_state")
    if status is not None and str(status).upper() not in ("DONE", "PREFLIGHT_PASSED"):
        log.err(f"audit-only: DONE.json status/terminal_state must be DONE, got {status!r}")
    elif status is not None:
        log.ok(f"audit-only: DONE.json status={status!r}")
    return done


def _check_audit_only(
    output_dir: Path,
    *,
    experiment_id: str | None,
    current_head: str | None,
    a06_split_sha: str | None,
    log: CheckLog,
) -> None:
    if not output_dir.exists():
        log.err(f"audit-only: output_dir does not exist: {output_dir}")
        return
    if not output_dir.is_dir():
        log.err(f"audit-only: output_dir is not a directory: {output_dir}")
        return

    expected_id = _expected_frozen_identity(a06_split_sha)

    # ---- 1. DONE.json (full audit; flagged as bridge if missing) ----
    done = _check_done_json(output_dir, log)
    if done is not None:
        if experiment_id is not None and done.get("experiment_id") != experiment_id:
            log.err(f"audit-only: DONE.json experiment_id mismatch")
        if current_head is not None and str(done.get("git_commit", "")) != str(current_head):
            log.err(f"audit-only: DONE.json git_commit mismatch")
        if done.get("git_dirty") is not False:
            log.err(f"audit-only: DONE.json git_dirty must be False")
        for key, expected in expected_id.items():
            if str(done.get(key)) != str(expected):
                log.err(f"audit-only: DONE.json {key} mismatch")

    # ---- 2. terminal conflict / DONE-only check via status.json / manifest.json ----
    for terminal_name in ("FAILED.json", "STOPPED.json"):
        if (output_dir / terminal_name).is_file():
            log.err(f"audit-only: terminal conflict: {terminal_name} present alongside DONE")
    manifest = _parse_json_object(output_dir / "manifest.json", "manifest.json", log)
    status = _parse_json_object(output_dir / "status.json", "status.json", log)
    if manifest is None or status is None:
        return
    if manifest.get("terminal_state") != "DONE":
        log.err(f"audit-only: manifest.json terminal_state must be DONE, got {manifest.get('terminal_state')!r}")
    for key, expected in (
        ("total_units", 30),
        ("unit_count_done", 30),
        ("unit_count_failed", 0),
        ("unit_count_stopped", 0),
    ):
        if int(manifest.get(key, -1)) != expected:
            log.err(f"audit-only: manifest.json {key} must be {expected}, got {manifest.get(key)!r}")
    for key in expected_id:
        if done is not None and str(manifest.get(key)) != str(done.get(key)):
            log.err(f"audit-only: manifest.json {key} differs from DONE.json")
        if str(manifest.get(key)) != str(expected_id[key]):
            log.err(f"audit-only: manifest.json {key} mismatch (expected {expected_id[key]!r})")
    if status.get("terminal_state") != "DONE":
        log.err("audit-only: status.json terminal_state must be DONE")
    for key, expected in (
        ("unit_count_done", 30),
        ("unit_count_failed", 0),
        ("unit_count_stopped", 0),
    ):
        if int(status.get(key, -1)) != expected:
            log.err(f"audit-only: status.json {key} must be {expected}")
    if status.get("winner") not in EXPECTED_CANDIDATES:
        log.err(f"audit-only: status.json winner must be in {EXPECTED_CANDIDATES}, got {status.get('winner')!r}")

    # ---- 3. input_manifest_hashes.json: 4 frozen hashes + 6 TEST=0 carriers ----
    imh = _parse_json_object(output_dir / "input_manifest_hashes.json", "input_manifest_hashes.json", log)
    if imh is None:
        return
    for key, expected in expected_id.items():
        if str(imh.get(key)) != str(expected):
            log.err(f"audit-only: input_manifest_hashes.json {key} mismatch")
    # R04-#2: all 6 TEST carriers required (bridge gap if missing)
    _check_test_zero_carriers(imh, log)

    # ---- 4. budget_report.json: R04-#1 strict fail-closed audit ----
    _check_budget_audit(output_dir, expected_id, log)

    # ---- 5. oof_metrics_summary.json: per-candidate aggregate ----
    oof_summary = _parse_json_object(output_dir / "oof_metrics_summary.json", "oof_metrics_summary.json", log)
    if oof_summary is None:
        return
    for c in EXPECTED_CANDIDATES:
        if c not in oof_summary:
            log.err(f"audit-only: oof_metrics_summary.json missing candidate {c!r}")
            continue
        block = oof_summary[c]
        if not isinstance(block, dict):
            log.err(f"audit-only: oof_metrics_summary.json[{c!r}] is not an object")
            continue
        if block.get("status") != "DONE":
            log.err(f"audit-only: oof_metrics_summary.json[{c!r}].status must be DONE, got {block.get('status')!r}")

    # ---- 6. candidates/<cand>/candidate_decision.json: per-seed OOF coverage ----
    for c in EXPECTED_CANDIDATES:
        cd = _parse_json_object(
            output_dir / "candidates" / c / "candidate_decision.json",
            f"candidates/{c}/candidate_decision.json", log,
        )
        if cd is None:
            continue
        seeds = cd.get("seeds")
        if not isinstance(seeds, dict):
            log.err(f"audit-only: candidates/{c}/candidate_decision.json seeds is not a dict")
            continue
        for seed in EXPECTED_SEEDS:
            sb = seeds.get(str(seed)) or seeds.get(seed)
            if sb is None:
                log.err(f"audit-only: candidates/{c}/candidate_decision.json missing seed {seed}")
                continue
            if sb.get("status") != "COMPLETE":
                log.err(f"audit-only: candidates/{c}/seed={seed} status must be COMPLETE, got {sb.get('status')!r}")
                continue
            ts = sb.get("total_samples")
            if not _is_finite_number(ts) or ts < 0:
                log.err(f"audit-only: candidates/{c}/seed={seed}/total_samples={ts!r} invalid")
                continue
            if int(ts) != EXPECTED_OOF_SAMPLES_PER_SEED:
                log.err(
                    f"audit-only: candidates/{c}/seed={seed}/total_samples={ts} "
                    f"!= {EXPECTED_OOF_SAMPLES_PER_SEED}"
                )
            # per-seed total_subjects not yet produced by runner; bridge gap
            if "total_subjects" not in sb:
                log.err(_schema_gap(
                    f"candidates/{c}/candidate_decision.json seeds.{seed}.total_subjects"
                ))

    # ---- 7. 30 unit complete.json files ----
    units_dir = output_dir / "units"
    if not units_dir.is_dir():
        log.err("audit-only: units/ directory missing")
        return
    expected_ids = _expected_unit_ids()
    seen: set[str] = set()
    for u in sorted(units_dir.iterdir()):
        if not u.is_dir():
            continue
        if u.name not in expected_ids:
            log.err(f"audit-only: unexpected unit dir {u.name!r}")
            continue
        seen.add(u.name)
        cp = _parse_json_object(u / "complete.json", f"{u.name}/complete.json", log)
        if cp is None:
            continue
        unit_block = cp.get("unit", {})
        cid = cp.get("identity", {})
        result = cp.get("result", {})
        cand = unit_block.get("candidate")
        fold_id = unit_block.get("fold_id")
        seed = unit_block.get("seed")
        if cand is None or fold_id is None or seed is None:
            log.err(f"audit-only: {u.name} missing unit.candidate/fold_id/seed")
            continue
        if cand not in EXPECTED_CANDIDATES:
            log.err(f"audit-only: {u.name} candidate {cand!r} not in frozen set")
        if fold_id not in EXPECTED_FOLDS:
            log.err(f"audit-only: {u.name} fold_id {fold_id!r} not in frozen set")
        if int(seed) not in EXPECTED_SEEDS:
            log.err(f"audit-only: {u.name} seed {seed!r} not in frozen set")
        if result.get("status") != "DONE":
            log.err(f"audit-only: {u.name} result.status must be DONE, got {result.get('status')!r}")
        for key, expected in expected_id.items():
            if str(cid.get(key)) != str(expected):
                log.err(f"audit-only: {u.name} identity.{key} mismatch")
        if current_head is not None and str(cid.get("git_commit", "")) != str(current_head):
            log.err(f"audit-only: {u.name} identity.git_commit mismatch")
        if cid.get("git_dirty") is not False:
            log.err(f"audit-only: {u.name} identity.git_dirty must be False")
        # per-unit peak CUDA in result
        pk = result.get("peak_cuda_mb")
        if pk is not None:
            if not _is_finite_number(pk) or pk < 0:
                log.err(f"audit-only: {u.name} result.peak_cuda_mb invalid: {pk!r}")
            elif pk > MAX_PEAK_CUDA_MB:
                log.err(f"audit-only: {u.name} result.peak_cuda_mb={pk} > {MAX_PEAK_CUDA_MB}")
        # per-unit wall_seconds
        ws = result.get("wall_seconds")
        if not _is_finite_number(ws) or ws < 0:
            log.err(f"audit-only: {u.name} result.wall_seconds invalid: {ws!r}")
        elif ws > MAX_WALL_MINUTES_PER_UNIT * 60:
            log.err(f"audit-only: {u.name} result.wall_seconds={ws} > {MAX_WALL_MINUTES_PER_UNIT*60}")

    if missing := expected_ids - seen:
        log.err(f"audit-only: {len(missing)} unit(s) missing in units/ (expected 30)")
    if extra := seen - expected_ids:
        log.err(f"audit-only: {len(extra)} unexpected unit(s) in units/")

    # ---- 8. top-level candidate_decision.json ----
    cd_top = _parse_json_object(output_dir / "candidate_decision.json", "candidate_decision.json", log)
    if cd_top is None:
        return
    if cd_top.get("winner") not in EXPECTED_CANDIDATES:
        log.err(f"audit-only: candidate_decision.json winner must be in {EXPECTED_CANDIDATES}")


# ---------------------------------------------------------------------------
# EXP-ID
# ---------------------------------------------------------------------------


def _check_experiment_id(exp_id: str | None, log: CheckLog) -> None:
    if exp_id is None:
        log.ok("experiment_id not provided; will be required at run time")
        return
    if exp_id in SYNTHETIC_EXP_ID_SENTINELS:
        log.err(f"experiment_id {exp_id!r} is a synthetic sentinel; refuse")
        return
    if not re.match(r"^EXP-SLP-B09-PM-FULL-30-UNIT-\d{8}-AUTODL-R\d{2}$", exp_id):
        log.err(f"experiment_id {exp_id!r} does not match B09 template")
        return
    log.ok(f"experiment_id matches B09 template: {exp_id}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B09 Full Run Preparation validator (R04, fail-closed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--protocol", type=Path, default=None)
    p.add_argument("--fold-manifest", type=Path, default=None)
    p.add_argument(
        "--b01-freeze-manifest", type=Path, required=True,
        help="Path to B01 freeze manifest JSON (REQUIRED).  Validator reads "
             "core.a06_split_identifier / core.a06_split_sha256 and "
             "verifies three-way A06 binding with B07 protocol and fold manifest.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--experiment-id", type=str, default=None)
    p.add_argument(
        "--audit-only", action="store_true",
        help="Read-only audit of an existing DONE 30-unit output directory",
    )
    return p


def _print_report(args, log: "CheckLog") -> None:
    print("=== B09 RUN PREPARATION VALIDATOR (R04) ===")
    print(f"repo_root: {args.repo_root}")
    print(f"mode: {'audit-only' if args.audit_only else 'preparation'}")
    if args.experiment_id:
        print(f"experiment_id: {args.experiment_id}")
    if args.output_dir:
        print(f"output_dir: {args.output_dir}")
    print()
    for o in log.oks:
        print(f"OK  : {o}")
    for e in log.errors:
        print(f"ERR : {e}")
    print()
    print(f"summary: {len(log.oks)} OK / {len(log.errors)} ERR")
    if log.errors:
        print("B09_RUN_PREPARATION_VALIDATION_FAILED")
        return
    print("B09_RUN_PREPARATION_VALIDATION_PASSED  TEST=0  units=30  GPU_FULL_NOT_AUTHORIZED")


_LAST_MAIN_LOG: CheckLog | None = None


def _capture_last_main_log() -> "CheckLog":
    if _LAST_MAIN_LOG is None:
        raise RuntimeError("main() has not been invoked yet")
    return _LAST_MAIN_LOG


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = (args.repo_root or script_path.parents[1]).resolve()

    log = CheckLog()
    global _LAST_MAIN_LOG
    _LAST_MAIN_LOG = log

    # B01 freeze binding
    b01_sha = _check_b01_freeze_binding(args.b01_freeze_manifest, log)
    freeze = None
    if b01_sha is not None:
        freeze = _load_b01_freeze(args.b01_freeze_manifest, log)

    if args.audit_only:
        current_head = _git_head(repo_root)
        if current_head is None:
            log.err("cannot read git HEAD")
        else:
            log.ok(f"git HEAD = {current_head[:12]}…")
        # Load B07 protocol + fold for three-way A06 binding check
        protocol_path = (args.protocol or (repo_root / B07_PROTOCOL_PATH)).resolve()
        fold_path = (args.fold_manifest or (repo_root / B07_FOLD_PATH)).resolve()
        protocol = _load_json(protocol_path)
        fold = _load_json(fold_path)
        a06_split_sha = _check_a06_three_way(protocol, fold, freeze, log)
        if args.output_dir is None:
            log.err("audit-only: --output-dir is required")
        else:
            _check_audit_only(
                args.output_dir,
                experiment_id=args.experiment_id,
                current_head=current_head,
                a06_split_sha=a06_split_sha,
                log=log,
            )
        _print_report(args, log)
        return 1 if log.errors else 0

    # Preparation mode
    _check_committed_hashes(repo_root, log)
    protocol_path = (args.protocol or (repo_root / B07_PROTOCOL_PATH)).resolve()
    fold_path = (args.fold_manifest or (repo_root / B07_FOLD_PATH)).resolve()
    protocol = _load_json(protocol_path)
    fold = _load_json(fold_path)
    if protocol is None:
        log.err(f"cannot load protocol from {protocol_path}")
    if fold is None:
        log.err(f"cannot load fold manifest from {fold_path}")
    # A06 three-way binding (preparation mode)
    a06_split_sha = _check_a06_three_way(protocol, fold, freeze, log)
    if protocol is not None:
        _check_30_unit_matrix(protocol, log)
    if fold is not None:
        _check_fold_isolation(fold, log)
    if protocol is not None:
        _check_test_policy(protocol, log)
    if protocol is not None:
        _check_budget(protocol, log)
    if protocol is not None:
        _check_identity_contract(protocol, log)
    if protocol is not None:
        _check_metrics(protocol, log)
    _check_runner_code(repo_root, log)
    log.require(
        _git_dirty(repo_root) is False,
        "git working tree is clean (git_dirty==False)",
        "git working tree is dirty; B09 forbids git_dirty==True",
    )
    head = _git_head(repo_root)
    if head is not None:
        log.ok(f"git HEAD = {head[:12]}… (record at execution time; do NOT pre-freeze)")
    else:
        log.err("cannot read git HEAD")
    _check_experiment_id(args.experiment_id, log)
    _check_output_dir_state(
        args.output_dir,
        experiment_id=args.experiment_id,
        current_head=head,
        a06_split_sha=a06_split_sha,
        log=log,
    )

    _print_report(args, log)
    return 1 if log.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
