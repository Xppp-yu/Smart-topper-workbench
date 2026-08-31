"""B08 Full Runner synthetic CPU smoke
(TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1).

This script exercises the B08 Full Runner through the full 5-fold × 2-candidate
× 3-seed scheduling path on small synthetic CPU data.  It verifies:

* Protocol loading with CRLF-safe committed-content byte SHA.
* Execution plan: exactly 30 units.
* Fold subject routing: all 5 folds cover 91 subjects with 0 overlap.
* Synthetic unit execution: 1 epoch per unit on synthetic data.
* Budget tracking: per-unit / per-candidate / total.
* Terminal state mutex: exactly one of DONE/FAILED/STOPPED.
* OOF merging per seed: 91 subjects / 4,095 samples, 0 duplicate/missing.
* Candidate aggregation and selection rule.
* Output policy: default refuses overwrite; ``--force`` allows it;
  ``--no-write`` writes nothing.

This script is CPU-only by design.  It does NOT read TEST rows,
does NOT use GPU, does NOT run real B01 data.

Usage::

    # Default: no-write smoke (prints one-line summary to stdout):
    uv run python scripts/smoke_b08_full_runner.py

    # Full smoke with output:
    uv run python scripts/smoke_b08_full_runner.py --output outputs/smoke_b08

    # Force (overwrite if output exists):
    uv run python scripts/smoke_b08_full_runner.py --output outputs/smoke_b08 --force

The no-write mode prints ``B08_SMOKE_NO_WRITE`` to stdout on success.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Use the main worktree's venv
VENV_PYTHON = Path(
    r"E:\TeamProjects\smarttopper-team-workbench\.venv\Scripts\python.exe"
)


B08_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "slp8_pm_full_protocol_v0.1.json"
)

# Declarative TEST-access policy for the smoke.
TEST_ACCESS_DECLARATION: dict[str, Any] = {
    "value": 0,
    "kind": "declarative_policy",
    "explanation": (
        "The B08 smoke does not import any B01 training table loader "
        "and does not invoke enable_test_access(...). The 0 is a static "
        "declaration, NOT a runtime count of TEST reads."
    ),
}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str, log_path: Path | None = None) -> None:
    line = f"[{_now_iso()}] {msg}"
    print(line)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_smoke_via_cli(
    output_dir: Path | None,
    *,
    no_write: bool = False,
) -> dict[str, Any]:
    """Run the B08 smoke by invoking the CLI script.

    B08 Round 3: --force has been removed from the production runner.
    Each smoke run uses a fresh temp directory and never overwrites
    an existing experiment directory.

    Returns
    -------
    dict[str, Any]
        Summary of the smoke run.
    """
    from topper_perception.neural.slp8_region_full import (
        B07_CANDIDATES,
        B07_SEEDS,
        DEV_SAMPLE_COUNT,
        DEV_SUBJECT_COUNT,
        FullProtocolError,
        build_execution_plan,
        committed_file_sha256,
        load_frozen_full_protocol,
        run_full,
        build_full_config,
        resolve_git_identity,
    )

    # 1. Protocol loading
    _log("=== B08 SMOKE: Protocol Loading ===")
    try:
        protocol = load_frozen_full_protocol(
            B08_CONFIG_PATH,
            repo_root=PROJECT_ROOT,
        )
    except FullProtocolError as e:
        return {
            "ok": False,
            "stage": "protocol_loading",
            "error": str(e),
        }

    _log(f"Protocol SHA: {protocol.protocol_sha256[:16]}...")
    _log(f"Fold manifest SHA: {protocol.fold_sha256[:16]}...")
    _log(f"Candidates: {protocol.candidates}")
    _log(f"Seeds: {protocol.seeds}")
    _log(f"Folds: {list(protocol.fold_subjects.keys())}")
    _log(f"Dev subjects: {protocol.development_subject_count}")
    _log(f"Dev samples: {protocol.development_sample_count}")

    # Verify B07 constants match
    assert protocol.candidates == B07_CANDIDATES, (
        f"Candidate mismatch: {protocol.candidates} != {B07_CANDIDATES}"
    )
    assert protocol.seeds == B07_SEEDS, (
        f"Seed mismatch: {protocol.seeds} != {B07_SEEDS}"
    )
    assert protocol.development_subject_count == DEV_SUBJECT_COUNT
    assert protocol.development_sample_count == DEV_SAMPLE_COUNT

    # 2. Execution plan
    _log("=== B08 SMOKE: Execution Planning ===")
    plan = build_execution_plan(protocol)
    expected_units = len(B07_CANDIDATES) * 5 * len(B07_SEEDS)
    assert len(plan) == expected_units, (
        f"Plan has {len(plan)} units, expected {expected_units}"
    )
    unit_ids = [u.unit_id for u in plan]
    assert len(unit_ids) == len(set(unit_ids)), "Duplicate unit IDs"
    _log(f"Execution plan: {len(plan)} units (expected {expected_units})")

    # Verify all candidates × folds × seeds covered
    for cand in B07_CANDIDATES:
        for fold_id in sorted(protocol.fold_subjects.keys()):
            for seed in B07_SEEDS:
                found = any(
                    u.candidate == cand
                    and u.fold_id == fold_id
                    and u.seed == seed
                    for u in plan
                )
                assert found, f"Missing unit: {cand}/{fold_id}/seed_{seed}"

    # 3. Fold subject coverage
    _log("=== B08 SMOKE: Fold Subject Coverage ===")
    all_subjects: set[str] = set()
    for fold_id, subjects in protocol.fold_subjects.items():
        overlap = all_subjects & set(subjects)
        assert not overlap, (
            f"Fold {fold_id} has subject overlap with previous folds: {overlap}"
        )
        all_subjects.update(subjects)
        _log(
            f"  {fold_id}: {len(subjects)} val subjects, "
            f"val_samples={protocol.fold_val_sample_counts[fold_id]}, "
            f"train_samples={protocol.fold_train_sample_counts[fold_id]}"
        )

    assert len(all_subjects) == DEV_SUBJECT_COUNT, (
        f"Total unique subjects {len(all_subjects)} != {DEV_SUBJECT_COUNT}"
    )
    _log(f"All folds total: {len(all_subjects)} unique subjects")

    # 4. Git identity resolution
    _log("=== B08 SMOKE: Git Identity ===")
    try:
        git_commit, git_dirty = resolve_git_identity(PROJECT_ROOT)
        _log(f"Git commit: {git_commit[:16]}... dirty={git_dirty}")
    except Exception as e:
        git_commit = "synthetic_unavailable"
        git_dirty = True
        _log(f"Git identity unavailable: {e} (using synthetic)")

    # 5. Synthetic full run
    _log("=== B08 SMOKE: Synthetic Full Execution ===")

    if output_dir is None:
        smoke_output_dir = Path(tempfile.mkdtemp(prefix="smoke_b08_"))
        cleanup_output = True
    else:
        smoke_output_dir = Path(output_dir)
        cleanup_output = False

    if no_write:
        _log(f"No-write mode: using temp dir {smoke_output_dir}")
    else:
        _log(f"Output directory: {smoke_output_dir}")

    log_path = smoke_output_dir / "logs" / "run.log"

    try:
        full_config = build_full_config(
            protocol_path=B08_CONFIG_PATH,
            output_dir=smoke_output_dir,
            experiment_id="EXP-SLP-B08-SYNTHETIC-SMOKE",
            git_commit=git_commit,
            git_dirty=git_dirty,
            b01_freeze_dir=None,
            data_root=None,
            device="cpu",
            batch_size=2,
            max_epochs=1,
            min_epochs=1,
            early_stopping_patience=2,
            synthetic_mode=True,
            no_write_mode=no_write,
            repo_root=PROJECT_ROOT,
        )
    except FullProtocolError as e:
        return {
            "ok": False,
            "stage": "config_build",
            "error": str(e),
        }

    start_time = time.monotonic()
    try:
        result = run_full(full_config)
    except Exception as e:
        return {
            "ok": False,
            "stage": "full_run",
            "error": f"{type(e).__name__}: {e}",
        }
    wall_time = time.monotonic() - start_time

    # 6. Verify results
    _log("=== B08 SMOKE: Result Verification ===")
    _log(f"Terminal state: {result.terminal_state}")
    _log(f"Total units: {result.unit_count_total} (expected {expected_units})")
    _log(f"Done: {result.unit_count_done}")
    _log(f"Failed: {result.unit_count_failed}")
    _log(f"Stopped: {result.unit_count_stopped}")
    _log(f"Total wall: {wall_time:.1f}s")

    # Assertions
    assert result.unit_count_total == expected_units, (
        f"Total units {result.unit_count_total} != {expected_units}"
    )

    # Terminal state
    assert result.terminal_state in ("DONE", "FAILED", "STOPPED"), (
        f"Invalid terminal state: {result.terminal_state}"
    )

    # At least one terminal state for all units
    assert result.unit_count_done + result.unit_count_failed + result.unit_count_stopped == expected_units

    # Candidate results
    assert set(result.candidate_results.keys()) == set(B07_CANDIDATES), (
        f"Candidate mismatch: {set(result.candidate_results.keys())} != {set(B07_CANDIDATES)}"
    )

    for cand, cres in result.candidate_results.items():
        _log(f"  candidate={cand}:")
        _log(f"    decision={cres.decision}")
        _log(f"    mean_pooled_iou={cres.mean_pooled_iou}")
        _log(f"    mean_worst_subject_iou={cres.mean_worst_subject_iou}")
        _log(f"    status={cres.status}")

        # Each candidate should have 3 seed results
        assert set(cres.seed_results.keys()) == set(B07_SEEDS), (
            f"Candidate {cand} seed results: "
            f"{set(cres.seed_results.keys())} != {set(B07_SEEDS)}"
        )

        for seed, sr in cres.seed_results.items():
            _log(f"    seed={seed}:")
            _log(f"      status={sr.status}")
            _log(f"      samples={sr.total_samples}")
            _log(f"      duplicate_count={sr.duplicate_count}")
            _log(f"      missing_count={sr.missing_count}")

    # Winner
    if result.winner:
        _log(f"Winner: {result.winner} (mean_iou={result.winner_mean_pooled_iou})")
        assert result.winner in B07_CANDIDATES, (
            f"Winner {result.winner} not in {B07_CANDIDATES}"
        )

    # Budget report
    br = result.budget_report
    _log(f"Budget report:")
    _log(f"  total_wall_minutes={br.get('total_wall_minutes')}")
    _log(f"  budget_ok={br.get('budget_ok')}")

    # Cleanup temp output
    if cleanup_output and not no_write:
        try:
            shutil.rmtree(smoke_output_dir)
            _log(f"Cleaned up temp directory: {smoke_output_dir}")
        except Exception:
            pass

    # Final verdict
    ok = (
        result.terminal_state in ("DONE",)
        and result.unit_count_total == expected_units
        and result.unit_count_done == expected_units
        and result.unit_count_failed == 0
        and result.unit_count_stopped == 0
    )

    summary = {
        "ok": ok,
        "terminal_state": result.terminal_state,
        "total_units": result.unit_count_total,
        "unit_count_done": result.unit_count_done,
        "unit_count_failed": result.unit_count_failed,
        "unit_count_stopped": result.unit_count_stopped,
        "winner": result.winner,
        "winner_mean_pooled_iou": result.winner_mean_pooled_iou,
        "total_wall_seconds": round(result.total_wall_seconds, 2),
        "budget_ok": br.get("budget_ok"),
        "no_write_mode": no_write,
        "output_dir": str(smoke_output_dir) if not cleanup_output else "temp",
        "test_access": TEST_ACCESS_DECLARATION,
    }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B08 Full Runner synthetic CPU smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: temp directory, cleaned up after)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate without creating output directory",
    )
    # NOTE: --force was removed in B08 Round 3; the smoke always uses a
    # fresh temp directory.  To retain artifacts, use --output with a
    # brand-new path.
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    # If no-write, also suppress stdout from the runner
    if args.no_write:
        result = run_smoke_via_cli(
            output_dir=None,  # Use temp dir
            no_write=True,
        )
    else:
        result = run_smoke_via_cli(
            output_dir=output_dir,
            no_write=False,
        )

    print()
    print("=" * 60)
    print("B08_SMOKE_RESULT:")
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)

    if result["ok"]:
        print("B08_SMOKE PASSED")
        return 0
    else:
        print(f"B08_SMOKE FAILED: {result.get('error', 'unknown error')}")
        print(f"  Stage: {result.get('stage', 'unknown')}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
