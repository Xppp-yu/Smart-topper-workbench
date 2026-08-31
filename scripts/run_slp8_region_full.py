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

    # Check authorization
    is_real_b01 = (b01_freeze_dir is not None)
    if is_real_b01 and not run_authorized:
        print("ERROR: real B01 run requires --run-authorized")
        return 1

    if is_real_b01 and run_authorized:
        if not experiment_id:
            print("ERROR: --experiment-id is required for real B01 runs")
            return 1
        if experiment_id == SYNTHETIC_EXP_ID:
            print(f"ERROR: --experiment-id cannot be the synthetic sentinel {SYNTHETIC_EXP_ID}")
            return 1
        if not Path(b01_freeze_dir).exists():
            print(f"ERROR: --b01-freeze-dir not found: {b01_freeze_dir}")
            return 1

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
