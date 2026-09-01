"""B08 Full Runner CLI (TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1).

Usage::

    # Validate-only (no-write):
    uv run python scripts/run_slp8_region_full.py \\
        --config configs/experiments/slp8_pm_full_protocol_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B08-PREFLIGHT-VALIDATE \\
        --validate-only

    # Synthetic CPU smoke:
    uv run python scripts/run_slp8_region_full.py \\
        --config configs/experiments/slp8_pm_full_protocol_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B08-SYNTH-SMOKE \\
        --synthetic-cpu-smoke

    # Real one-fold preflight (NOT executed by this task; requires separate
    # Owner authorization):
    uv run python scripts/run_slp8_region_full.py \\
        --config configs/experiments/slp8_pm_full_protocol_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B08-PREFLIGHT-R01 \\
        --b01-freeze-dir data/processed/slp8_training_tables_v0.1 \\
        --dataset-root <SLP8_DATASET_ROOT> \\
        --run-authorized \\
        --experiment-id EXP-SLP-B08-PREFLIGHT-R01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp8_training_table_freeze import (
    A06_SPLIT_SHA256_EXPECTED,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SOURCE_SPLITS,
    EXPECTED_SETTINGS,
    EXPECTED_COVERS,
    sha256_file,
)
from topper_perception.neural.slp8_region_full import (
    B07_CANDIDATES,
    B07_CONFIG_VERSION,
    B07_FOLD_CONFIG_VERSION,
    B07_PROTOCOL_NAME,
    B07_SEEDS,
    B08_TASK_ID,
    DEV_SAMPLE_COUNT,
    DEV_SUBJECT_COUNT,
    FullConfigValidationError,
    FullExperimentIdentityError,
    FullOutputCollisionError,
    FullProtocolError,
    FullRunAuthorizationError,
    FullUnit,
    SYNTHETIC_EXP_ID,
    SYNTHETIC_SMOKE_DEFAULTS,
    build_execution_plan,
    build_model,
    build_full_config,
    committed_file_sha256,
    file_sha256,
    load_frozen_full_protocol,
    load_real_b01_fold,
    load_checkpoint_for_resume,
    Slp8RegionDataset,
    build_dataloader,
    refuse_overwrite,
    resolve_git_identity,
    train_one_unit,
    run_full,
)


# ---------------------------------------------------------------------------
# B09 CLI bridge constants
# ---------------------------------------------------------------------------

#: Required EXP-ID template for B09 30-unit real-B01 entry.
B09_EXP_ID_REGEX = r"^EXP-SLP-B09-PM-FULL-30-UNIT-\d{8}-AUTODL-R\d{2}$"


#: Sentinel B09 synthetic EXP-IDs that must never be allowed for --run-full.
B09_SYNTHETIC_SENTINELS: frozenset[str] = frozenset({
    SYNTHETIC_EXP_ID,
    "EXP-SLP-B09-SYNTHETIC-SMOKE",
})


#: Frozen training contract values that --run-full must not allow
#: callers to override.  Sources:
#:   - B07 protocol: ``configs/experiments/slp8_pm_full_protocol_v0.1.json``
#:     (max_epochs / min_epochs / early_stopping_patience / seeds /
#:      candidates / folds / total_units / resource_budget).
#:   - B09 preparation §15: per-unit batch_size 16 (runner default,
#:     matches B08 R03).
B09_FROZEN_MAX_EPOCHS: int = 30
B09_FROZEN_MIN_EPOCHS: int = 5
B09_FROZEN_EARLY_STOPPING_PATIENCE: int = 4
B09_FROZEN_BATCH_SIZE: int = 16
B09_FROZEN_MAX_WALL_MINUTES_PER_UNIT: int = 15
B09_FROZEN_MAX_PEAK_CUDA_MB: int = 8192
B09_FROZEN_TOTAL_UNITS: int = 30


#: Strict 40-character lowercase hexadecimal SHA pattern.
B09_GIT_SHA_REGEX = r"^[0-9a-f]{40}$"


#: Terminal file names that seal a B09 experiment and must not be
#: overwritten by a fresh ``--run-full`` dispatch.  Anything else
#: inside ``output_dir`` is treated as resumable intermediate state
#: and is forwarded to ``run_full()`` for identity / per-unit /
#: checkpoint / budget-state verification.
B09_SEALED_TERMINAL_NAMES: tuple[str, ...] = ("DONE.json", "FAILED.json", "STOPPED.json")
B09_RESUME_IDENTITY_FILENAME = "resume_identity.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj):
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_default(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _log(msg: str, log_path: Path | None = None) -> None:
    ts = _now_iso()
    line = f"[{ts}] {msg}"
    print(line)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _log_reject(msg: str) -> None:
    """B09 bridge rejection log: print to stdout ONLY; never touch the filesystem.

    The bridge contract requires that all authorization / parameter /
    EXP-ID / git / protocol / identity / frozen-contract / terminal
    / output-collision gates complete before any experiment directory
    is created.  Rejection therefore uses this helper instead of
    :func:`_log`, which would otherwise ``mkdir(parents=True)`` the
    experiment's ``logs/`` tree.
    """
    ts = _now_iso()
    print(f"[{ts}] REJECT: {msg}", file=sys.stderr)


def _canonical_resume_identity(payload: dict, *, nested: bool = False) -> dict:
    """Return the frozen experiment identity from a persisted carrier."""
    identity = payload.get("identity") if nested else payload
    if not isinstance(identity, dict):
        raise FullProtocolError("resume identity carrier has no JSON object identity")
    return {
        "experiment_id": identity.get("experiment_id", identity.get("exp_id")),
        "git_commit": identity.get("git_commit"),
        "config_sha256": identity.get("config_sha256"),
        "data_manifest_sha256": identity.get("data_manifest_sha256"),
        "fold_manifest_sha256": identity.get("fold_manifest_sha256"),
        "split_sha256": identity.get(
            "split_sha256", identity.get("a06_split_sha256")
        ),
    }


def _read_resume_identity_carrier(path: Path, *, nested: bool = False) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullProtocolError(f"resume identity carrier {path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FullProtocolError(f"resume identity carrier {path} is not a JSON object")
    return _canonical_resume_identity(payload, nested=nested)


def _validate_b09_partial_resume_identity(
    output_dir: Path,
    expected_identity: dict,
) -> bool:
    """Fail closed before touching a non-terminal partial experiment.

    Returns ``True`` when an existing partial directory was validated and
    ``False`` for a fresh/empty output directory. Every persisted identity
    carrier that is present must match; a non-empty directory with no identity
    carrier is not a governed resume and is rejected.
    """
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return False

    carriers: list[tuple[Path, bool]] = []
    for name in (B09_RESUME_IDENTITY_FILENAME, "manifest.json", "status.json"):
        path = output_dir / name
        if path.is_file():
            carriers.append((path, False))
    budget_path = output_dir / "budget_state.json"
    if budget_path.is_file():
        carriers.append((budget_path, True))
    carriers.extend(
        (path, True) for path in sorted((output_dir / "units").glob("*/complete.json"))
    )

    if not carriers:
        raise FullProtocolError(
            f"non-terminal output directory {output_dir} has no governed "
            "resume identity carrier"
        )

    for path, nested in carriers:
        actual = _read_resume_identity_carrier(path, nested=nested)
        for key, expected_value in expected_identity.items():
            if actual.get(key) != expected_value:
                raise FullProtocolError(
                    f"resume identity mismatch in {path} on {key!r}: "
                    f"expected {expected_value!r}, got {actual.get(key)!r}"
                )
    return True


def _write_b09_resume_identity_atomic(output_dir: Path, identity: dict) -> None:
    """Persist an early identity carrier so first-unit interruption is resumable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / B09_RESUME_IDENTITY_FILENAME
    if path.is_file():
        return
    payload = {**identity, "git_dirty": False, "written_at_utc": _now_iso()}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SLP8 B07/B08 Full Runner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the frozen B07 protocol JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output directory",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate protocol without writing output or reading data",
    )
    p.add_argument(
        "--synthetic-cpu-smoke",
        action="store_true",
        help="Run synthetic CPU smoke (validate-only with scheduling semantics)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Validate and plan without creating output directory or artifacts",
    )
    # NOTE: --force was removed in B08 Round 3; the production runner
    # refuses to overwrite any existing experiment directory.  Synthetic
    # smoke runs use a fresh temporary directory instead.
    p.add_argument(
        "--b01-freeze-dir",
        type=Path,
        default=None,
        help="Path to B01 freeze tables (required for real runs)",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="SLP8 dataset root (passed to B01 loader)",
    )
    p.add_argument(
        "--run-authorized",
        action="store_true",
        help=(
            "REQUIRED for real B01 runs.  Explicit Owner authorization "
            "to execute training on real data."
        ),
    )
    p.add_argument(
        "--one-fold-preflight",
        action="store_true",
        help="Run exactly one real B01 fold/candidate/seed preflight; never runs the 30-unit Full plan",
    )
    # B09 CLI bridge (TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1):
    # exclusive with --one-fold-preflight / --validate-only / --no-write /
    # --synthetic-cpu-smoke.  Requires --run-authorized, real --b01-freeze-dir,
    # --dataset-root, --experiment-id matching B09_EXP_ID_REGEX and a clean
    # committed worktree.  This is the only entry point that calls
    # run_full() with synthetic_mode=False, no_write_mode=False.
    p.add_argument(
        "--run-full",
        action="store_true",
        help=(
            "B09: execute the entire 30-unit real B01 Full plan by "
            "constructing the resolved FullConfig from the frozen B07 "
            "protocol and dispatching it to run_full() exactly once. "
            "Requires --run-authorized, --b01-freeze-dir, --dataset-root, "
            "--experiment-id and a clean committed worktree; refuses "
            "without explicit Owner authorization."
        ),
    )
    p.add_argument("--candidate", choices=list(B07_CANDIDATES), default=None)
    p.add_argument("--fold-id", choices=[f"fold_{i}" for i in range(1, 6)], default=None)
    p.add_argument("--seed", type=int, choices=list(B07_SEEDS), default=None)
    p.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Owner-supplied EXP-ID (required for real runs)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training (default: cuda)",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max training epochs (default: 30)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Git repository root (default: auto-detect from config path)",
    )
    return p


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------


def run_validate_only(
    config: Path,
    output_dir: Path,
    *,
    repo_root: Path,
) -> int:
    """Validate the B07 protocol and output plan to stdout — ZERO files written.

    This function creates NO output files, NO directories, and NO logs.
    All validation results are printed to stdout only.
    """
    print(f"=== VALIDATE-ONLY MODE (no-write) ===")
    print(f"Config: {config}")
    print(f"Output dir (will NOT be created): {output_dir}")

    # Verify output_dir does not exist (no-write contract)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        # Check if any important artifact exists
        for artifact in ["manifest.json", "status.json", "DONE.json",
                         "FAILED.json", "STOPPED.json", "resolved_config.json"]:
            if (output_dir / artifact).exists():
                raise FullOutputCollisionError(
                    f"validate-only mode requires output_dir {output_dir} to not exist "
                    f"(found {artifact}); use a different directory"
                )

    print(f"Loading frozen B07 protocol from {config}...")
    protocol = load_frozen_full_protocol(config, repo_root=repo_root)

    print(f"Protocol SHA: {protocol.protocol_sha256}")
    print(f"Fold manifest SHA: {protocol.fold_sha256}")
    print(f"Candidates: {protocol.candidates}")
    print(f"Seeds: {protocol.seeds}")
    print(f"Folds: {list(protocol.fold_subjects.keys())}")
    print(f"Development subjects: {protocol.development_subject_count}")
    print(f"Development samples: {protocol.development_sample_count}")

    plan = build_execution_plan(protocol)
    print(f"Execution plan: {len(plan)} units")

    for unit in plan:
        print(f"  {unit.unit_id}")

    print(f"VALIDATE_ONLY PASSED: units={len(plan)}, TEST=0")
    return 0


def run_one_fold_preflight(
    *, config: Path, output_dir: Path, repo_root: Path, b01_freeze_dir: Path,
    dataset_root: Path, experiment_id: str, candidate: str,
    fold_id: str, seed: int, device: str, batch_size: int,
    max_epochs: int,
) -> int:
    """Execute exactly one governed real-data fold/candidate/seed unit.

    The preflight deliberately bypasses :func:`run_full`: it trains only the
    explicitly selected unit, records measured wall/CUDA budget, verifies the
    checkpoint identity and an independent double reload prediction hash, and
    writes a self-contained preflight bundle.  It never builds the 30-unit
    execution plan and never enables B01 TEST access.
    """
    if not experiment_id or experiment_id == SYNTHETIC_EXP_ID:
        raise FullExperimentIdentityError("one-fold preflight requires a non-synthetic EXP-ID")
    protocol = load_frozen_full_protocol(config, repo_root=repo_root)
    if candidate not in protocol.candidates:
        raise FullProtocolError(f"candidate {candidate!r} is not in frozen B07 protocol")
    if seed not in protocol.seeds:
        raise FullProtocolError(f"seed {seed!r} is not in frozen B07 protocol")
    if fold_id not in protocol.fold_subjects:
        raise FullProtocolError(f"fold {fold_id!r} is not in frozen B07 protocol")
    raw_protocol = json.loads(config.read_text(encoding="utf-8"))
    frozen_training = raw_protocol["training_contract"]
    if max_epochs != int(frozen_training["max_epochs"]):
        raise FullProtocolError(
            "one-fold preflight must use the frozen B07 max_epochs="
            f"{frozen_training['max_epochs']}, got {max_epochs}"
        )
    git_commit, git_dirty = resolve_git_identity(repo_root)
    if git_dirty:
        raise FullExperimentIdentityError(
            "real one-fold preflight requires a clean committed worktree"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FullOutputCollisionError(
            f"one-fold preflight output directory is not empty: {output_dir}"
        )
    full_config = build_full_config(
        protocol_path=config,
        output_dir=output_dir,
        experiment_id=experiment_id,
        git_commit=git_commit,
        git_dirty=False,
        b01_freeze_dir=b01_freeze_dir,
        data_root=dataset_root,
        device=device,
        batch_size=batch_size,
        max_epochs=max_epochs,
        min_epochs=int(frozen_training["min_epochs"]),
        early_stopping_patience=int(frozen_training["early_stopping_patience"]),
        synthetic_mode=False,
        repo_root=repo_root,
    )
    val_subjects = protocol.fold_subjects[fold_id]
    train_s, val_s, norm, cw = load_real_b01_fold(
        b01_freeze_dir, dataset_root, fold_id, val_subjects, synthetic_mode=False
    )
    if norm is None or cw is None or not train_s or not val_s:
        raise FullProtocolError("one-fold preflight requires non-empty real TRAIN/VAL and fitted preprocessing")
    unit = FullUnit(candidate=candidate, fold_id=fold_id, seed=seed)
    unit_dir = output_dir / "unit"
    identity = {
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "git_dirty": False,
        "config_sha256": full_config.config_sha256,
        "data_manifest_sha256": full_config.data_manifest_sha256,
        "fold_manifest_sha256": full_config.fold_manifest_sha256,
        "split_sha256": full_config.a06_split_sha256,
        "model_version": candidate,
        "candidate": candidate,
        "fold_id": fold_id,
        "seed": seed,
    }
    try:
        result = train_one_unit(
            unit=unit,
            train_records=train_s,
            val_records=val_s,
            config=full_config,
            unit_output_dir=unit_dir,
            normalization=norm,
            class_weight_result=cw,
            data_root=dataset_root,
            val_sample_ids=[s.sample_id for s in val_s],
            val_subject_ids_list=[s.subject_id for s in val_s],
            val_postures=[s.posture for s in val_s],
        )
    except Exception as exc:
        payload = {
            **identity,
            "status": "FAILED",
            "unit_status": "FAILED",
            "error": f"train_one_unit failed: {type(exc).__name__}: {exc}",
            "test_access": False,
        }
        _write_json(output_dir / "preflight_manifest.json", payload)
        _write_json(output_dir / "FAILED.json", payload)
        return 1
    if result.status != "DONE" or result.checkpoint_best_path is None:
        payload = {
            **identity, "status": "FAILED", "unit_status": result.status,
            "error": result.error_message, "test_access": False,
        }
        _write_json(output_dir / "preflight_manifest.json", payload)
        _write_json(output_dir / "FAILED.json", payload)
        return 1

    def _reload_prediction_hash() -> str:
        resumed = load_checkpoint_for_resume(result.checkpoint_best_path, identity)
        model = build_model(candidate, "cpu")
        model.load_state_dict(resumed["model_state_dict"])
        model.eval()
        loader = build_dataloader(
            Slp8RegionDataset(
                samples=val_s, dataset_root=dataset_root, normalization=norm,
            ),
            batch_size=batch_size, shuffle=False, drop_last=False,
        )
        digest = hashlib.sha256()
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["pressure"].to("cpu")).argmax(dim=1)
                digest.update(np.ascontiguousarray(pred.numpy()).tobytes())
        return digest.hexdigest()

    try:
        first_hash = _reload_prediction_hash()
        second_hash = _reload_prediction_hash()
    except Exception as exc:
        payload = {
            **identity, "status": "FAILED", "unit_status": result.status,
            "error": f"checkpoint reload audit failed: {type(exc).__name__}: {exc}",
            "wall_seconds": result.wall_seconds,
            "peak_cuda_mb": result.peak_cuda_mb,
            "test_access": False,
        }
        _write_json(output_dir / "preflight_manifest.json", payload)
        _write_json(output_dir / "FAILED.json", payload)
        return 1
    reload_consistent = first_hash == second_hash
    within_wall_budget = (
        result.wall_seconds <= full_config.max_wall_minutes_per_unit * 60
    )
    within_cuda_budget = (
        result.peak_cuda_mb is not None
        and result.peak_cuda_mb <= full_config.max_peak_cuda_mb
    )
    passed = reload_consistent and within_wall_budget and within_cuda_budget
    status = "PREFLIGHT_PASSED" if passed else "FAILED"
    payload = {
        **identity, "status": status,
        "candidate": candidate, "fold_id": fold_id, "seed": seed,
        "train_samples": len(train_s), "val_samples": len(val_s),
        "train_subjects": len({s.subject_id for s in train_s}),
        "val_subjects": len({s.subject_id for s in val_s}),
        "device": device, "wall_seconds": result.wall_seconds,
        "peak_cuda_mb": result.peak_cuda_mb,
        "max_wall_minutes": full_config.max_wall_minutes_per_unit,
        "max_peak_cuda_mb": full_config.max_peak_cuda_mb,
        "best_epoch": result.best_epoch,
        "best_checkpoint_sha256": file_sha256(result.checkpoint_best_path),
        "reload_prediction_hash": first_hash,
        "reload_consistent": reload_consistent,
        "within_wall_budget": within_wall_budget,
        "within_cuda_budget": within_cuda_budget,
        "test_access": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "preflight_manifest.json", payload)
    _write_json(output_dir / ("DONE.json" if passed else "FAILED.json"), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# B09 CLI bridge: --run-full entry (TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1)
# ---------------------------------------------------------------------------


def _check_run_full_mutex(
    args: argparse.Namespace,
    log_path: Path,
) -> tuple[bool, str]:
    """Verify the B09 --run-full mutex contract.

    Returns (ok, error_message).  Logs all decisions to ``log_path`` so
    the audit trail is preserved if the caller later proceeds.
    """
    mutex_failures: list[str] = []

    if getattr(args, "one_fold_preflight", False):
        mutex_failures.append("--run-full cannot be combined with --one-fold-preflight")
    if getattr(args, "validate_only", False):
        mutex_failures.append("--run-full cannot be combined with --validate-only")
    if getattr(args, "no_write", False):
        mutex_failures.append("--run-full cannot be combined with --no-write")
    if getattr(args, "synthetic_cpu_smoke", False):
        mutex_failures.append("--run-full cannot be combined with --synthetic-cpu-smoke")

    if mutex_failures:
        return False, "; ".join(mutex_failures)

    # Required real B01 inputs.
    if not getattr(args, "b01_freeze_dir", None):
        mutex_failures.append("--b01-freeze-dir is required for --run-full")
    if not getattr(args, "dataset_root", None):
        mutex_failures.append("--dataset-root is required for --run-full")
    if not getattr(args, "run_authorized", False):
        mutex_failures.append("--run-authorized is required for --run-full")
    if not getattr(args, "experiment_id", None):
        mutex_failures.append("--experiment-id is required for --run-full")

    if mutex_failures:
        return False, "; ".join(mutex_failures)

    return True, ""


def run_full_b09(
    config: Path,
    output_dir: Path,
    *,
    repo_root: Path,
    b01_freeze_dir: Path,
    dataset_root: Path,
    experiment_id: str,
    device: str,
    batch_size: int,
    max_epochs: int,
) -> int:
    """B09 CLI bridge: dispatch the 30-unit real B01 Full plan.

    The function performs every fail-closed gate required by the B09
    bridge contract, then constructs a single ``FullConfig`` and calls
    :func:`run_full` exactly once.  No training loop is duplicated.

    R02 invariants:

    * Every rejection path uses :func:`_log_reject`, which prints to
      stderr but does NOT touch the filesystem.  The experiment
      directory is created only after all gates have passed and the
      resolved :class:`FullConfig` has been built.
    * Sealed-terminal directories (DONE.json / FAILED.json /
      STOPPED.json) are refused; non-terminal partial output is
      forwarded to :func:`run_full` which performs the existing
      per-unit / identity / checkpoint / budget-state resume
      verification.
    * The CLI-supplied ``batch_size`` must equal the frozen
      ``B09_FROZEN_BATCH_SIZE``; any drift is rejected before
      ``output_dir`` is created.

    Gates (all must pass before ``output_dir`` is created and before
    any training begins):

    1. EXP-ID must match ``B09_EXP_ID_REGEX``; synthetic sentinels are
       rejected.
    2. Worktree must be a clean committed git tree
       (``git_dirty=False``); the resolved commit must be a strict
       40-character lowercase hex SHA.
    3. ``output_dir`` must not contain a sealed terminal artifact
       (DONE.json / FAILED.json / STOPPED.json).  Any other
       intermediate state is forwarded to ``run_full`` for resume.
    4. Frozen B07 protocol must load; the resolved protocol must use
       exactly the frozen training contract values
       (max_epochs, min_epochs, patience, candidates, seeds, folds,
       total_units, resource_budget).
    5. ``batch_size`` must equal ``B09_FROZEN_BATCH_SIZE`` (16).
    6. ``build_full_config`` must succeed with
       ``synthetic_mode=False`` and ``load_test=False`` (enforced
       via the real B01 path inside ``build_full_config``).
    7. ``run_full()`` is invoked exactly once with
       ``synthetic_mode=False`` and ``no_write_mode=False``; the
       function is forbidden from re-parsing, mutating, or
       duplicating the 30-unit plan.
    """
    output_dir = Path(output_dir).resolve()
    b01_freeze_dir = Path(b01_freeze_dir).resolve()
    dataset_root = Path(dataset_root).resolve()

    # ------------------------------------------------------------------
    # Gate 1: EXP-ID must match B09 template; reject synthetic sentinels.
    # ------------------------------------------------------------------
    if experiment_id in B09_SYNTHETIC_SENTINELS:
        _log_reject(
            f"--run-full refuses synthetic EXP-ID {experiment_id!r}"
        )
        return 2
    if not re.match(B09_EXP_ID_REGEX, experiment_id):
        _log_reject(
            f"--experiment-id {experiment_id!r} does not match "
            f"required template {B09_EXP_ID_REGEX}"
        )
        return 2

    # ------------------------------------------------------------------
    # Gate 2: Git identity must be a clean, parseable 40-char hex SHA.
    # ------------------------------------------------------------------
    try:
        git_commit, git_dirty = resolve_git_identity(repo_root)
    except FullExperimentIdentityError as exc:
        _log_reject(f"git identity unresolvable: {exc}")
        return 2
    if git_dirty:
        _log_reject(
            "--run-full requires a clean committed worktree; "
            f"git_dirty=True (commit={git_commit})"
        )
        return 2
    if not isinstance(git_commit, str) or not re.fullmatch(
        B09_GIT_SHA_REGEX, git_commit
    ):
        _log_reject(
            f"--run-full requires a 40-char lowercase hex SHA, "
            f"got {git_commit!r}"
        )
        return 2

    # ------------------------------------------------------------------
    # Gate 3: Output dir must not already have a sealed terminal state.
    # Non-terminal partial output is forwarded to run_full() for
    # resume; the runner's own per-unit / identity / budget state
    # verification (load_resume_state + atomic complete.json) is
    # the single source of truth.
    # ------------------------------------------------------------------
    for terminal_name in B09_SEALED_TERMINAL_NAMES:
        if (output_dir / terminal_name).is_file():
            _log_reject(
                f"--run-full refuses to overwrite sealed terminal "
                f"{terminal_name} in {output_dir}"
            )
            return 3

    # ------------------------------------------------------------------
    # Gate 4: Load frozen B07 protocol and verify frozen values match
    # the CLI / bridge contract.
    # ------------------------------------------------------------------
    try:
        protocol = load_frozen_full_protocol(config, repo_root=repo_root)
    except FullProtocolError as exc:
        _log_reject(f"protocol load failed: {exc}")
        return 2

    raw_protocol = json.loads(config.read_text(encoding="utf-8"))
    frozen_training = raw_protocol["training_contract"]
    frozen_budget = raw_protocol["resource_budget"]
    frozen_matrix = raw_protocol["execution_matrix"]

    if int(frozen_training["max_epochs"]) != B09_FROZEN_MAX_EPOCHS:
        _log_reject(
            f"frozen B07 max_epochs={frozen_training['max_epochs']} "
            f"!= bridge constant {B09_FROZEN_MAX_EPOCHS}"
        )
        return 2
    if max_epochs != B09_FROZEN_MAX_EPOCHS:
        _log_reject(
            f"--max-epochs={max_epochs} does not match frozen "
            f"B07 value {B09_FROZEN_MAX_EPOCHS}"
        )
        return 2
    if int(frozen_training["min_epochs"]) != B09_FROZEN_MIN_EPOCHS:
        _log_reject(
            f"frozen B07 min_epochs={frozen_training['min_epochs']} "
            f"!= bridge constant {B09_FROZEN_MIN_EPOCHS}"
        )
        return 2
    if int(frozen_training["early_stopping_patience"]) != B09_FROZEN_EARLY_STOPPING_PATIENCE:
        _log_reject(
            f"frozen B07 early_stopping_patience="
            f"{frozen_training['early_stopping_patience']} != bridge constant "
            f"{B09_FROZEN_EARLY_STOPPING_PATIENCE}"
        )
        return 2
    if set(tuple(frozen_training["seeds"])) != set(protocol.seeds):
        _log_reject(
            f"frozen B07 seeds {frozen_training['seeds']} differ from "
            f"resolved protocol seeds {protocol.seeds}"
        )
        return 2
    if int(frozen_matrix.get("candidates", 0)) != len(protocol.candidates):
        _log_reject(
            f"frozen execution_matrix.candidates="
            f"{frozen_matrix.get('candidates')} differs from "
            f"resolved candidates={len(protocol.candidates)}"
        )
        return 2
    if int(frozen_matrix.get("folds", 0)) != len(protocol.fold_subjects):
        _log_reject(
            f"frozen execution_matrix.folds={frozen_matrix.get('folds')} "
            f"differs from resolved folds={len(protocol.fold_subjects)}"
        )
        return 2
    if int(frozen_matrix.get("total_units", 0)) != B09_FROZEN_TOTAL_UNITS:
        _log_reject(
            f"frozen execution_matrix.total_units="
            f"{frozen_matrix.get('total_units')} is not "
            f"{B09_FROZEN_TOTAL_UNITS}"
        )
        return 2
    if int(frozen_budget.get("max_wall_minutes_per_fold_seed_unit", -1)) != B09_FROZEN_MAX_WALL_MINUTES_PER_UNIT:
        _log_reject(
            f"frozen resource_budget.max_wall_minutes_per_fold_seed_unit="
            f"{frozen_budget.get('max_wall_minutes_per_fold_seed_unit')} != "
            f"{B09_FROZEN_MAX_WALL_MINUTES_PER_UNIT}"
        )
        return 2
    if int(frozen_budget.get("max_peak_cuda_mb", -1)) != B09_FROZEN_MAX_PEAK_CUDA_MB:
        _log_reject(
            f"frozen resource_budget.max_peak_cuda_mb="
            f"{frozen_budget.get('max_peak_cuda_mb')} != "
            f"{B09_FROZEN_MAX_PEAK_CUDA_MB}"
        )
        return 2

    # ------------------------------------------------------------------
    # Gate 5: Frozen CLI batch_size (= 16, B08 R03 default + B09 §15).
    # ------------------------------------------------------------------
    if batch_size != B09_FROZEN_BATCH_SIZE:
        _log_reject(
            f"--batch-size={batch_size} does not match frozen "
            f"B09 value {B09_FROZEN_BATCH_SIZE}"
        )
        return 2

    # ------------------------------------------------------------------
    # Gate 6: Build the resolved FullConfig.  This step does not write
    # to ``output_dir``; it only reads the protocol and freeze manifest
    # and constructs an in-memory FullConfig.  If it fails, the bridge
    # exits without ever creating the experiment directory.
    # ------------------------------------------------------------------
    try:
        full_config = build_full_config(
            protocol_path=config,
            output_dir=output_dir,
            experiment_id=experiment_id,
            git_commit=git_commit,
            git_dirty=False,
            b01_freeze_dir=b01_freeze_dir,
            data_root=dataset_root,
            device=device,
            batch_size=batch_size,
            max_epochs=B09_FROZEN_MAX_EPOCHS,
            min_epochs=B09_FROZEN_MIN_EPOCHS,
            early_stopping_patience=B09_FROZEN_EARLY_STOPPING_PATIENCE,
            synthetic_mode=False,
            no_write_mode=False,
            repo_root=repo_root,
        )
    except (FullProtocolError, FullConfigValidationError) as exc:
        _log_reject(f"build_full_config failed: {exc}")
        return 2

    expected_resume_identity = {
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "config_sha256": full_config.config_sha256,
        "data_manifest_sha256": full_config.data_manifest_sha256,
        "fold_manifest_sha256": full_config.fold_manifest_sha256,
        "split_sha256": full_config.a06_split_sha256,
    }
    try:
        _validate_b09_partial_resume_identity(output_dir, expected_resume_identity)
    except FullProtocolError as exc:
        _log_reject(f"partial resume identity rejected: {exc}")
        return 4

    # ------------------------------------------------------------------
    # All gates passed.  Now it is safe to create the experiment
    # directory tree and start the structured run log.
    # ------------------------------------------------------------------
    _write_b09_resume_identity_atomic(output_dir, expected_resume_identity)
    log_path = output_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(
        f"=== B09 --run-full BRIDGE ===",
        log_path,
    )
    _log(f"experiment_id={experiment_id}", log_path)
    _log(f"git_commit={git_commit} dirty=False", log_path)
    _log(f"output_dir={output_dir}", log_path)
    _log(f"b01_freeze_dir={b01_freeze_dir}", log_path)
    _log(f"dataset_root={dataset_root}", log_path)
    _log(f"device={device} batch_size={batch_size}", log_path)
    _log(
        f"frozen candidates={list(protocol.candidates)} "
        f"seeds={list(protocol.seeds)} "
        f"folds={list(protocol.fold_subjects.keys())}",
        log_path,
    )

    # ------------------------------------------------------------------
    # Dispatch run_full() exactly once.  This is the single source of
    # truth for the 30-unit plan; no parallel writer may be added by
    # the bridge contract.  run_full() handles interruption / resume
    # (load_resume_state + atomic complete.json) and identity drift.
    # ------------------------------------------------------------------
    try:
        result = run_full(full_config)
    except FullOutputCollisionError as exc:
        _log(f"OUTPUT COLLISION: {exc}", log_path)
        return 4
    except FullProtocolError as exc:
        _log(f"PROTOCOL ERROR: {exc}", log_path)
        return 4
    except FullRunAuthorizationError as exc:
        _log(f"AUTHORIZATION ERROR: {exc}", log_path)
        return 4
    except Exception as exc:
        _log(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", log_path)
        traceback.print_exc()
        return 4

    # ------------------------------------------------------------------
    # Summary report (mirrors the B08 synthetic smoke output style).
    # ------------------------------------------------------------------
    print("B09_RUN_FULL_RESULT:")
    print(f"  terminal_state: {result.terminal_state}")
    print(f"  total_units: {result.unit_count_total}")
    print(f"  unit_count_done: {result.unit_count_done}")
    print(f"  unit_count_failed: {result.unit_count_failed}")
    print(f"  unit_count_stopped: {result.unit_count_stopped}")
    print(f"  total_wall_seconds: {round(result.total_wall_seconds, 2)}")
    print(f"  winner: {result.winner}")
    print(f"  winner_mean_pooled_iou: {result.winner_mean_pooled_iou}")
    print(f"  budget_ok: {result.budget_report.get('budget_ok')}")
    print(f"  config_sha256: {result.config_sha256}")
    print(f"  data_manifest_sha256: {result.data_manifest_sha256}")
    print(f"  fold_manifest_sha256: {result.fold_manifest_sha256}")
    print(f"  a06_split_sha256: {result.a06_split_sha256}")

    if result.terminal_state == "DONE" and result.unit_count_done == result.unit_count_total:
        return 0
    if result.terminal_state == "FAILED":
        return 5
    if result.terminal_state == "STOPPED":
        return 6
    return 4


# ---------------------------------------------------------------------------
# Synthetic CPU smoke mode
# ---------------------------------------------------------------------------


def run_synthetic_cpu_smoke(
    config: Path,
    output_dir: Path,
    *,
    repo_root: Path,
) -> int:
    """Run synthetic CPU smoke with full scheduling semantics.

    B08 Round 5: ``--force`` has been removed from the production CLI.
    The smoke uses a fresh temporary directory by default; if ``--output``
    is provided, the caller is responsible for using a brand-new path.
    """
    _log(f"=== SYNTHETIC CPU SMOKE MODE ===", None)
    _log(f"Config: {config}", None)
    _log(f"Output: {output_dir}", None)

    output_dir = Path(output_dir).resolve()

    log_path = output_dir / "logs" / "run.log"

    # Resolve git identity
    try:
        git_commit, git_dirty = resolve_git_identity(repo_root)
    except FullExperimentIdentityError:
        git_commit = "synthetic_git_unavailable"
        git_dirty = True

    _log(f"Git commit: {git_commit} dirty={git_dirty}", log_path)

    # Build configuration
    try:
        full_config = build_full_config(
            protocol_path=config,
            output_dir=output_dir,
            experiment_id=SYNTHETIC_EXP_ID,
            git_commit=git_commit,
            git_dirty=git_dirty,
            b01_freeze_dir=None,
            data_root=None,
            device="cpu",
            batch_size=2,
            max_epochs=SYNTHETIC_SMOKE_DEFAULTS["max_epochs_per_unit"],
            min_epochs=SYNTHETIC_SMOKE_DEFAULTS["min_epochs"],
            early_stopping_patience=SYNTHETIC_SMOKE_DEFAULTS["early_stopping_patience"],
            synthetic_mode=True,
            no_write_mode=False,
            repo_root=repo_root,
        )
    except (FullProtocolError, FullConfigValidationError) as e:
        _log(f"CONFIG ERROR: {e}", log_path)
        return 1

    # Execute
    _log("Starting synthetic full run...", log_path)
    start = time.monotonic()

    try:
        result = run_full(full_config)
    except FullOutputCollisionError as e:
        _log(f"OUTPUT COLLISION: {e}", log_path)
        print(f"B08_SMOKE FAILED: {e}")
        return 1
    except FullProtocolError as e:
        _log(f"PROTOCOL ERROR: {e}", log_path)
        print(f"B08_SMOKE FAILED: {e}")
        return 1
    except Exception as e:
        _log(f"UNEXPECTED ERROR: {type(e).__name__}: {e}", log_path)
        traceback.print_exc()
        return 1

    wall = time.monotonic() - start
    _log(f"Synthetic run completed in {wall:.1f}s", log_path)

    # Print summary
    print(f"B08_SMOKE_RESULT:")
    print(f"  terminal_state: {result.terminal_state}")
    print(f"  total_units: {result.unit_count_total}")
    print(f"  unit_count_done: {result.unit_count_done}")
    print(f"  unit_count_failed: {result.unit_count_failed}")
    print(f"  unit_count_stopped: {result.unit_count_stopped}")
    print(f"  total_wall_seconds: {round(result.total_wall_seconds, 2)}")
    print(f"  winner: {result.winner}")
    print(f"  winner_mean_pooled_iou: {result.winner_mean_pooled_iou}")
    print(f"  budget_ok: {result.budget_report.get('budget_ok')}")

    # Per-candidate summary
    for cand, cres in result.candidate_results.items():
        print(f"  candidate={cand}:")
        print(f"    decision: {cres.decision}")
        print(f"    mean_pooled_iou: {cres.mean_pooled_iou}")
        print(f"    mean_worst_subject_iou: {cres.mean_worst_subject_iou}")
        print(f"    status: {cres.status}")
        for seed, sr in cres.seed_results.items():
            print(f"    seed={seed}: status={sr.status}, samples={sr.total_samples}")

    # Terminal state check
    terminal_path = output_dir / f"{result.terminal_state}.json"
    print(f"  terminal_file: {terminal_path.name}")

    if result.terminal_state == "DONE" and result.unit_count_done == result.unit_count_total:
        print(f"B08_SMOKE PASSED")
        return 0
    else:
        print(f"B08_SMOKE INCOMPLETE: state={result.terminal_state}")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    config = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not config.is_file():
        print(f"ERROR: config not found: {config}")
        return 1

    # Detect repo root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = config.parents[2]  # configs/experiments/../../ = project root

    # Detect mode
    no_write = getattr(args, "no_write", False)
    synthetic = getattr(args, "synthetic_cpu_smoke", False)
    validate_only = getattr(args, "validate_only", False)
    run_authorized = getattr(args, "run_authorized", False)
    b01_freeze_dir = getattr(args, "b01_freeze_dir", None)
    experiment_id = getattr(args, "experiment_id", None)
    one_fold = getattr(args, "one_fold_preflight", False)
    run_full_flag = getattr(args, "run_full", False)

    # Check authorization
    is_real_b01 = (b01_freeze_dir is not None)
    if is_real_b01 and not run_authorized and not run_full_flag:
        # Existing preflight contract: refusing real B01 without
        # --run-authorized.  The --run-full branch performs its own
        # exhaustive authorization check below; keep this short-circuit
        # for one-fold to avoid mutating its semantic.
        print("ERROR: real B01 run requires --run-authorized")
        return 1

    if is_real_b01 and run_authorized and not run_full_flag:
        if not experiment_id:
            print("ERROR: --experiment-id is required for real B01 runs")
            return 1
        if experiment_id == SYNTHETIC_EXP_ID:
            print(f"ERROR: --experiment-id cannot be the synthetic sentinel {SYNTHETIC_EXP_ID}")
            return 1
        if not Path(b01_freeze_dir).exists():
            print(f"ERROR: --b01-freeze-dir not found: {b01_freeze_dir}")
            return 1

    if run_full_flag:
        # B09 CLI bridge entry point.  The bridge performs every
        # fail-closed gate (mutex, EXP-ID regex, git identity, frozen
        # contract, output dir, then a single run_full() dispatch).
        mutex_ok, mutex_msg = _check_run_full_mutex(args, Path(output_dir) / "logs" / "run.log")
        if not mutex_ok:
            print(f"ERROR: {mutex_msg}")
            return 2
        if not Path(b01_freeze_dir).exists():
            print(f"ERROR: --b01-freeze-dir not found: {b01_freeze_dir}")
            return 2
        if not Path(args.dataset_root).exists():
            print(f"ERROR: --dataset-root not found: {args.dataset_root}")
            return 2
        try:
            return run_full_b09(
                config=config,
                output_dir=output_dir,
                repo_root=repo_root,
                b01_freeze_dir=Path(b01_freeze_dir),
                dataset_root=Path(args.dataset_root),
                experiment_id=experiment_id,
                device=args.device,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
            )
        except (FullProtocolError, FullExperimentIdentityError,
                FullConfigValidationError, FullRunAuthorizationError) as e:
            print(f"ERROR: {e}")
            return 2

    if one_fold:
        if not (is_real_b01 and run_authorized and args.dataset_root and experiment_id):
            print("ERROR: --one-fold-preflight requires real B01, --dataset-root, --experiment-id and --run-authorized")
            return 1
        if not args.candidate or not args.fold_id or args.seed is None:
            print("ERROR: --one-fold-preflight requires --candidate, --fold-id and --seed")
            return 1
        try:
            return run_one_fold_preflight(
                config=config, output_dir=output_dir, repo_root=repo_root,
                b01_freeze_dir=Path(b01_freeze_dir), dataset_root=Path(args.dataset_root),
                experiment_id=experiment_id, candidate=args.candidate,
                fold_id=args.fold_id, seed=args.seed, device=args.device,
                batch_size=args.batch_size, max_epochs=args.max_epochs,
            )
        except (FullProtocolError, FullExperimentIdentityError) as e:
            print(f"ERROR: {e}")
            return 1

    # Validate-only / no-write mode: ZERO files written
    if validate_only or no_write:
        try:
            return run_validate_only(
                config=config,
                output_dir=output_dir,
                repo_root=repo_root,
            )
        except (FullProtocolError, FullOutputCollisionError) as e:
            print(f"ERROR: {e}")
            return 1

    # Synthetic smoke mode (writes outputs)
    if synthetic:
        if is_real_b01:
            print("ERROR: --synthetic-cpu-smoke cannot be combined with --b01-freeze-dir")
            return 1
        try:
            return run_synthetic_cpu_smoke(
                config=config,
                output_dir=output_dir,
                repo_root=repo_root,
            )
        except (FullProtocolError, FullOutputCollisionError,
                FullExperimentIdentityError, FullConfigValidationError) as e:
            print(f"ERROR: {e}")
            return 1

    # Real B01 run (NOT executed by this task)
    if is_real_b01 and run_authorized:
        print("ERROR: Real B01 run not executed by this task.")
        print("  Real preflight requires separate Owner authorization.")
        return 1

    # Default: validate-only
    try:
        return run_validate_only(
            config=config,
            output_dir=output_dir,
            repo_root=repo_root,
        )
    except (FullProtocolError, FullOutputCollisionError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
