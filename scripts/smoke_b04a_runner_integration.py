"""B04A runner-integration smoke (TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1).

This script drives the B04A architecture-expansion runner through the
3 candidates x 3 seeds orchestration path on a small synthetic
CPU dataset.  It exercises:

* protocol dispatch (B04 vs B04A via ``config_version``);
* the B04A validator (active set, forbidden candidates, exact
  three seeds, threshold, TEST=0, lifecycle);
* the B04A multi-seed orchestrator (per-seed identity, per-seed
  budget accumulator, ``all_seeds_must_succeed``);
* the 0/1/2/3-feasible candidate-level decision rule;
* the near-tie tiebreak (prefer simpler model);
* identity carrier format (JSON, CSV sibling, log first line,
  checkpoint identity);
* terminal-state mutex (DONE / FAILED / STOPPED);
* output policy (default refuses overwrite; ``--force`` allows
  it; ``--no-write`` writes nothing).

The script is CPU-only by design and never reads TEST rows.  It
follows the same conventions as
``scripts/smoke_b04a_implementation.py``: a single
``B04A_SMOKE_NO_WRITE`` line on stdout under ``--no-write``, and
a structured JSON summary otherwise.

This script is **not** a real Mini run.  It only validates the
B04A runner-integration contract.  No GPU, no B01 freeze tables,
no real data, no TEST rows, no ranking claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topper_perception.neural.slp8_region_mini import (  # noqa: E402
    B04A_ACTIVE_CANDIDATE_NAMES,
    B04A_CONFIG_VERSION,
    B04A_FEASIBILITY_THRESHOLD,
    B04A_FORBIDDEN_CANDIDATE_NAMES,
    B04A_NEAR_TIE_MARGIN,
    B04A_PROTOCOL_NAME,
    B04A_SEEDS,
    B04_PROTOCOL_NAME,
    ConfigValidationError,
    MiniProtocolError,
    SYNTHETIC_EXP_ID,
    _b04a_advance_decision,
    _b04a_aggregate_candidate,
    _b04a_identity_block,
    _b04a_per_region_pass,
    _b04a_seed_class_collapse,
    _b04a_worst_subject_pass,
    _compute_synthetic_manifest_sha256,
    _protocol_of_config,
    _write_b04a_run_bundle,
    build_mini_config,
    run_mini_b04a,
    validate_mini_config,
)


B04A_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "slp8_pm_architecture_expansion_mini_v0.1.json"
)
B04_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "slp8_pm_region_mini_v0.1.json"
)


# Declarative TEST-access policy.  This is NOT a runtime counter;
# the script does not import any B01 test-access contract.
TEST_ACCESS_DECLARATION: dict[str, Any] = {
    "value": 0,
    "kind": "declarative_policy",
    "explanation": (
        "The B04A runner-integration smoke does not import any B01 "
        "training table loader and does not invoke "
        "enable_test_access(...). The 0 is a static declaration, NOT a "
        "runtime count of TEST reads."
    ),
}


# ---------------------------------------------------------------------------
# Synthetic CPU mini-rig (pure stdlib + torch)
# ---------------------------------------------------------------------------


def _build_synthetic_b04a(
    n_train: int = 4, n_val: int = 2, seed: int = 42
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Build a tiny synthetic dataset + TRAIN-only class stats.

    The synthetic path mirrors :func:`build_synthetic_dataset` from
    :mod:`topper_perception.neural.slp8_region_mini` but is inlined so
    this script does not have to import the B04 mini helper (which
    forces a small initial import surface).  The output is
    functionally equivalent for the B04A smoke: every class 0..8 is
    present at least once in TRAIN.
    """

    from topper_perception.neural.slp8_region_mini import (
        PRESSURE_SHAPE, N_CLASSES, Slp8SyntheticDataset,
        build_synthetic_dataset,
    )
    train_dataset, val_dataset, dataset_manifest, train_class_stats = (
        build_synthetic_dataset(
            n_train_samples=n_train,
            n_val_samples=n_val,
            seed=seed,
        )
    )
    return train_dataset, val_dataset, dataset_manifest, train_class_stats


def _smoke_run_one(
    config_path: Path,
    output_dir: Path,
    *,
    no_write: bool = False,
    force: bool = False,
    budget_override_seconds: float | None = None,
) -> dict[str, Any]:
    """Run a single B04A synthetic CPU smoke and return a structured summary.

    When ``no_write=True`` the function does NOT call the B04A
    orchestrator's bundle writer; it only exercises the orchestrator
    itself and reports the result.  The orchestrator still needs an
    output directory for its per-seed checkpoint path, so the
    no-write mode routes the orchestrator to a temporary directory
    that is cleaned up before the function returns.
    """

    from topper_perception.neural.slp8_region_mini import (
        ResourceBudget,
        _gather_environment,
        _write_b04a_run_bundle,
        assert_class_weight_invariants,
        compute_class_weights,
        file_sha256,
        resolve_device,
    )
    import torch

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    if _protocol_of_config(raw) != B04A_PROTOCOL_NAME:
        raise MiniProtocolError(
            f"smoke_b04a_runner_integration: config_version "
            f"{raw.get('config_version')!r} is not the B04A protocol; "
            f"refusing to run the B04A smoke"
        )

    # Validate the frozen protocol config first; then apply
    # synthetic-CPU overrides.  The B04A protocol mandates
    # ``device=cuda`` so the validator would otherwise reject the
    # ``cpu`` override.
    # Synthetic override: force CPU, tiny per-candidate epoch budget.
    raw["training"]["device"] = "cpu"
    raw["training"]["max_epochs"] = 1
    raw["training"]["min_epochs"] = 1
    raw["training"]["early_stopping"]["patience"] = 1
    config = build_mini_config(
        raw,
        b01_freeze_dir="<SYNTHETIC>",
        data_root="<SYNTHETIC>",
        config_path=str(config_path),
    )
    if config.protocol != B04A_PROTOCOL_NAME:
        raise MiniProtocolError(
            f"build_mini_config produced protocol {config.protocol!r}; "
            f"expected {B04A_PROTOCOL_NAME!r}"
        )
    if tuple(config.seeds) != B04A_SEEDS:
        raise MiniProtocolError(
            f"MiniConfig.seeds {config.seeds!r} != B04A_SEEDS "
            f"{B04A_SEEDS!r}"
        )
    if set(config.candidates) != set(B04A_ACTIVE_CANDIDATE_NAMES):
        raise MiniProtocolError(
            f"MiniConfig.candidates {config.candidates!r} != "
            f"B04A_ACTIVE_CANDIDATE_NAMES {list(B04A_ACTIVE_CANDIDATE_NAMES)!r}"
        )

    device = resolve_device("cpu", allow_cpu_fallback=True)
    if str(device) != "cpu":
        raise MiniProtocolError("B04A smoke must run on cpu")

    if budget_override_seconds is not None and budget_override_seconds > 0:
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=float(budget_override_seconds),
            max_wall_seconds_total=float(budget_override_seconds) * 3,
            max_peak_cuda_mb=8192.0,
        )
    else:
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=45 * 60.0,
            max_wall_seconds_total=135 * 60.0,
            max_peak_cuda_mb=8192.0,
        )

    train_ds, val_ds, dataset_manifest, train_class_stats = _build_synthetic_b04a(
        n_train=4, n_val=2, seed=42
    )
    class_weight_result = compute_class_weights(
        {
            "n_samples": int(train_class_stats["n_samples"]),
            "n_pixels": int(train_class_stats["n_pixels"]),
            "per_class_pixel_ratio": {
                int(k): float(v)
                for k, v in train_class_stats["per_class_pixel_ratio"].items()
            },
        }
    )
    assert_class_weight_invariants(class_weight_result)

    # R05 ITERATE: resolve Git identity once here, the same way the
    # CLI does.  The smoke exercises the full carrier pipeline, so the
    # per-seed CheckpointIdentity blocks must receive the same value
    # the CLI would freeze at dispatch time.
    from topper_perception.neural.slp8_region_mini import (
        _resolve_git_identity,
    )

    smoke_git_commit, smoke_git_dirty = _resolve_git_identity()

    started_at = time.time()
    # In no-write mode, the orchestrator still needs a writable
    # directory for its per-seed checkpoint path.  Use a
    # temporary directory so the no-write run does not collide
    # with previous outputs and does not leave anything on disk.
    if no_write:
        orchestrator_output_dir = Path(tempfile.mkdtemp(prefix="b04a_smoke_"))
        try:
            result = run_mini_b04a(
                config=config,
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataset_manifest=dataset_manifest,
                class_weight_result=class_weight_result,
                output_dir=orchestrator_output_dir,
                device=device,
                input_hashes={
                    "config_sha256": file_sha256(config_path),
                    "a06_split_sha256_expected": str(
                        config.b01_a06_split_sha256_expected
                    ),
                    "synthetic": True,
                },
                train_class_stats_source="synthetic_train_class_stats",
                synthetic=True,
                budget=budget,
                experiment_id=SYNTHETIC_EXP_ID,
                data_manifest_sha256=_compute_synthetic_manifest_sha256(),
                git_commit=smoke_git_commit,
                git_dirty=smoke_git_dirty,
            )
        finally:
            shutil.rmtree(orchestrator_output_dir, ignore_errors=True)
    else:
        result = run_mini_b04a(
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            dataset_manifest=dataset_manifest,
            class_weight_result=class_weight_result,
            output_dir=output_dir,
            device=device,
            input_hashes={
                "config_sha256": file_sha256(config_path),
                "a06_split_sha256_expected": str(
                    config.b01_a06_split_sha256_expected
                ),
                "synthetic": True,
            },
            train_class_stats_source="synthetic_train_class_stats",
            synthetic=True,
            budget=budget,
            experiment_id=SYNTHETIC_EXP_ID,
            data_manifest_sha256=_compute_synthetic_manifest_sha256(),
            git_commit=smoke_git_commit,
            git_dirty=smoke_git_dirty,
        )
    ended_at = time.time()
    result.wall_clock_seconds = float(ended_at - started_at)

    summary: dict[str, Any] = {
        "task_id": config.task_id,
        "stage": "S2_B04A",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "config_version": config.config_version,
        "protocol": config.protocol,
        "seeds": list(config.seeds),
        "candidates": list(config.candidates),
        "forbidden_candidates": list(B04A_FORBIDDEN_CANDIDATE_NAMES),
        "feasibility_threshold": B04A_FEASIBILITY_THRESHOLD,
        "near_tie_margin": B04A_NEAR_TIE_MARGIN,
        "terminal_state": result.terminal_state,
        "overall_decision": result.overall_decision,
        "advanced": list(result.advanced),
        "near_tie_applied": bool(result.near_tie_applied),
        "advance_decision": result.advance_decision.as_dict(),
        "n_candidates_feasible": int(result.n_candidates_feasible),
        "n_candidates_not_feasible": int(result.n_candidates_not_feasible),
        "n_candidates_failed": int(result.n_candidates_failed),
        "n_candidates_stopped": int(result.n_candidates_stopped),
        "wall_clock_seconds": float(result.wall_clock_seconds),
        "candidate_feasibility": {
            cand: {
                "feasibility": agg.feasibility,
                "reason": agg.reason,
                "macro_iou_mean": agg.macro_iou_mean,
                "n_seeds_total": agg.n_seeds_total,
                "n_seeds_feasible": agg.n_seeds_feasible,
                "n_seeds_failed": agg.n_seeds_failed,
                "n_seeds_stopped": agg.n_seeds_stopped,
                "elapsed_seconds_total": float(agg.elapsed_seconds_total),
                "budget_status": agg.budget_status,
            }
            for cand, agg in result.candidate_results.items()
        },
        "test_access": TEST_ACCESS_DECLARATION,
        "identity_keys": sorted(
            _b04a_identity_block(
                config=config,
                config_sha256=file_sha256(config_path),
                experiment_id=SYNTHETIC_EXP_ID,
                data_manifest_sha256=_compute_synthetic_manifest_sha256(),
                git_commit=result.git_commit,
                git_dirty=result.git_dirty,
            ).keys()
        ),
        "environment": _gather_environment(),
        "notes": [
            "B01 training tables, TEST rows, and TEST labels are NEVER "
            "loaded by this smoke script.",
            "The B04A smoke is CPU-only and never instantiates a CUDA "
            "context.",
            "Synthetic per-candidate epoch budget = 1 (protocol budget "
            "is 30); the runner-integration smoke only proves the "
            "orchestration path, not the protocol's training budget.",
        ],
    }

    if not no_write:
        config_sha256 = file_sha256(config_path)
        _write_b04a_run_bundle(
            output_dir=output_dir,
            result=result,
            config_sha256=config_sha256,
        )

    return summary


def _check_output_path(path: Path, *, force: bool) -> None:
    """Refuse to overwrite an existing populated output directory.

    Mirrors the B04 runner's ``check_output_dir_safety`` policy:
    any existing ``DONE.json`` / ``FAILED.json`` / ``STOPPED.json``,
    or any non-``.gitkeep`` file in the directory, blocks the smoke
    unless ``force=True``.
    """

    if not path.exists():
        return
    if not path.is_dir():
        raise MiniProtocolError(
            f"output path exists but is not a directory: {path}"
        )
    sentinels = ("DONE.json", "FAILED.json", "STOPPED.json")
    for s in sentinels:
        if (path / s).is_file():
            raise MiniProtocolError(
                f"output directory already contains {s}; refusing to "
                f"overwrite.  Choose a fresh output_dir or pass --force.  "
                f"({path})"
            )
    contents = [p for p in path.iterdir() if p.name != ".gitkeep"]
    if contents and not force:
        raise MiniProtocolError(
            f"output directory is not empty ({len(contents)} entries); "
            f"refusing to overwrite.  Choose a fresh output_dir or pass "
            f"--force.  ({path})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=B04A_CONFIG_PATH,
        help=(
            "Path to the B04A architecture expansion Mini config.  "
            "Defaults to the frozen B04A config; the smoke refuses any "
            "non-B04A config_version."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "b04a_runner_integration_smoke_v0.1.json",
        help=(
            "Path to write the JSON summary.  Refuses to overwrite an "
            "existing file unless --force is set.  Ignored under --no-write."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments" / "_b04a_runner_integration_smoke",
        help=(
            "Run-level output directory for the B04A bundle "
            "(manifest, status, candidate_decision, budget_report, "
            "per-seed checkpoints).  Ignored under --no-write."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow overwriting the existing JSON summary file and the "
            "existing run-level output directory.  Use with care."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Run the smoke pipeline but do NOT write any file.  A "
            "single-line summary is printed to stdout; the B04A "
            "orchestrator's bundle writer is NOT invoked."
        ),
    )
    parser.add_argument(
        "--budget-override-seconds",
        type=float,
        default=None,
        help=(
            "Test-only knob: override the per-candidate wall budget "
            "(seconds) to drive the STOPPED state.  The B04A protocol "
            "budget remains 2700 seconds; this option is for "
            "fail-closed tests only."
        ),
    )
    args = parser.parse_args(argv)

    if args.no_write:
        summary = _smoke_run_one(
            args.config,
            args.output_dir,
            no_write=True,
            budget_override_seconds=args.budget_override_seconds,
        )
        n_cpu_candidates = len(summary["candidates"])
        n_seeds = len(summary["seeds"])
        n_feasible = summary["n_candidates_feasible"]
        n_failed = summary["n_candidates_failed"]
        n_stopped = summary["n_candidates_stopped"]
        any_seed_feasible = any(
            info["n_seeds_feasible"] > 0
            for info in summary["candidate_feasibility"].values()
        )
        all_seeds_attempted = all(
            info["n_seeds_total"] == n_seeds
            and info["n_seeds_failed"] == 0
            and info["n_seeds_stopped"] == 0
            for info in summary["candidate_feasibility"].values()
        )
        print(
            f"B04A_SMOKE_NO_WRITE protocol={summary['protocol']} "
            f"config_version={summary['config_version']} "
            f"candidates={n_cpu_candidates} seeds={n_seeds} "
            f"terminal_state={summary['terminal_state']} "
            f"feasible={n_feasible} failed={n_failed} stopped={n_stopped} "
            f"advanced={len(summary['advanced'])} "
            f"near_tie_applied={summary['near_tie_applied']} "
            f"all_seeds_attempted={bool(all_seeds_attempted)} "
            f"any_seed_feasible={bool(any_seed_feasible)}"
        )
        return 0

    if args.output.exists() and not args.force:
        print(
            f"ERROR: output file already exists: {args.output}.  "
            f"Refusing to overwrite.  Pass --force to allow overwrite, "
            f"or pass --output to a different path.",
            file=sys.stderr,
        )
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        print(
            f"ERROR: output directory is not empty: {args.output_dir}.  "
            f"Refusing to overwrite.  Pass --force to allow overwrite, "
            f"or pass --output-dir to a different path.",
            file=sys.stderr,
        )
        return 2
    if args.output_dir.exists() and args.force:
        # Force: nuke existing output directory and recreate.
        shutil.rmtree(args.output_dir, ignore_errors=True)

    try:
        summary = _smoke_run_one(
            args.config,
            args.output_dir,
            no_write=False,
            force=args.force,
            budget_override_seconds=args.budget_override_seconds,
        )
    except ConfigValidationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except MiniProtocolError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2

    # Write the mutually exclusive terminal file (DONE/FAILED/STOPPED)
    # in the same output directory as the B04A bundle.  The CLI
    # contract is one terminal file per run.
    from topper_perception.neural.slp8_region_mini import (
        write_status_files,
    )
    write_status_files(
        args.output_dir,
        status=summary["terminal_state"],
        extra={
            "task_id": summary["task_id"],
            "protocol": summary["protocol"],
            "config_version": summary["config_version"],
            "mode": "synthetic-cpu-smoke-b04a",
            "overall_decision": summary["overall_decision"],
            "advanced": summary["advanced"],
            "near_tie_applied": summary["near_tie_applied"],
            "wall_clock_seconds": summary["wall_clock_seconds"],
            "n_candidates_feasible": summary["n_candidates_feasible"],
            "n_candidates_not_feasible": summary["n_candidates_not_feasible"],
            "n_candidates_failed": summary["n_candidates_failed"],
            "n_candidates_stopped": summary["n_candidates_stopped"],
            "terminal_state": summary["terminal_state"],
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0 if summary["terminal_state"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
