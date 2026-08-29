"""B04 PM-only Region Mini runner CLI (TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1).

This script is the B04 entry point.  It enforces the B04 governance:

* The default mode is ``--validate-config``, which reads the frozen
  config and writes ``status.json``/``resolved_config.json``/``DONE.json``
  without touching any B01 data.
* The synthetic CPU smoke mode is invoked with ``--synthetic-cpu-smoke``
  and uses tiny deterministic synthetic (pressure, label) pairs that
  fully exercise the runner, the registry, the class-weight formula,
  the metrics bundle, the checkpoint save/load, and the FEASIBLE gate.
* A real B01 run requires **both** ``--run-authorized`` and
  ``--b01-freeze-dir`` / ``--dataset-root``; the script refuses to
  read any real B01 path when ``--run-authorized`` is missing.  The
  current task never exercises the real B01 path.

Usage (default = --validate-config)::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-VALIDATE

Synthetic CPU smoke::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH \\
        --synthetic-cpu-smoke

Real run (NOT executed by B04 v0.1; requires explicit --run-authorized)::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-R01 \\
        --b01-freeze-dir <B01_FREEZE_DIR> \\
        --dataset-root <SLP8_DATASET_ROOT> \\
        --run-authorized
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from topper_perception.io.slp8_training_table_freeze import (
    A06_SPLIT_SHA256_EXPECTED,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SOURCE_SPLITS,
    EXPECTED_SETTINGS,
    EXPECTED_COVERS,
    load_b01_freeze_tables,
    sha256_file,
)
from topper_perception.neural.slp8_region_b01_contract import (
    B01FreezeSnapshot,
    build_b01_contract_expected,
    check_freeze_manifest_file_consistency,
    verify_b01_contract,
)
from topper_perception.neural.slp8_region_class_weights import (
    assert_class_weight_invariants,
    compute_class_weights,
)
from topper_perception.neural.slp8_region_dataset import (
    verify_subject_isolation,
)
from topper_perception.neural.slp8_region_budget import (
    ResourceBudget,
    resource_budget_from_config,
)
from topper_perception.neural.slp8_region_mini import (
    B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
    B04_CANDIDATE_NAMES,
    B04_PROTOCOL_NAME,
    B04A_ACTIVE_CANDIDATE_NAMES,
    B04A_CONFIG_VERSION,
    B04A_FEASIBILITY_THRESHOLD,
    B04A_FORBIDDEN_CANDIDATE_NAMES,
    B04A_PROTOCOL_NAME,
    B04A_SEEDS,
    B04A_TASK_ID,
    CHECKPOINT_VERSION,
    MINI_VERSION,
    SYNTHETIC_DEFAULTS,
    SYNTHETIC_DEFAULTS_B04A,
    TASK_ID,
    MiniConfig,
    MiniProtocolError,
    OutputCollisionError,
    ResourceBudget,
    _gather_environment,
    _protocol_of_config,
    _write_b04a_run_bundle,
    build_mini_config,
    build_synthetic_dataset,
    check_output_dir_safety,
    file_sha256,
    resolve_device,
    resource_budget_from_config,
    run_mini,
    run_mini_b04a,
    validate_mini_config,
    write_mini_artifacts,
    write_status_files,
)
from topper_perception.neural.slp8_region_models import (
    B04_MAX_PARAMETERS,
    B04A_EXACT_PARAMETER_COUNTS,
    B04A_MAX_PARAMETERS,
    DEEPLABV3PLUS_LITE_VERSION,
    MODEL_VERSION,
    RESUNET_LITE_VERSION,
    SMALL_UNET_VERSION,
    get_model_builder,
)
from topper_perception.neural.slp8_region_resume import (
    ResumeRefusedError,
    refuse_resume_for_done_run,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDACTED_LOCAL_PATH = "REDACTED_LOCAL_PATH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------


def _run_validate_config(
    config_path: Path, output_dir: Path
) -> int:
    """Validate the config (and the model registry) without running anything.

    This is a protocol-dispatch entry point.  After the shared config
    validation, the resolved ``MiniConfig.protocol`` routes to the
    B04 branch (writing the historical B04 identity) or the B04A
    branch (writing the frozen B04A protocol identity).  Cross-
    protocol use is fail-closed: an unknown protocol raises
    :class:`MiniProtocolError` before any artifact is written.
    """

    output_dir = Path(output_dir).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    _log("config validation: PASSED")

    # Build a MiniConfig purely to serialize the resolved view and
    # to drive the protocol dispatch.
    config = build_mini_config(
        raw,
        b01_freeze_dir=None,
        data_root=None,
        config_path=str(config_path),
    )
    _log(f"resolved protocol={config.protocol!r} config_version={config.config_version!r}")

    if config.protocol == B04_PROTOCOL_NAME:
        return _run_validate_config_b04(
            config_path=config_path,
            output_dir=output_dir,
            raw=raw,
            config=config,
            log_path=log_path,
        )
    if config.protocol == B04A_PROTOCOL_NAME:
        return _run_validate_config_b04a(
            config_path=config_path,
            output_dir=output_dir,
            raw=raw,
            config=config,
            log_path=log_path,
        )
    raise MiniProtocolError(
        f"_run_validate_config: config.protocol={config.protocol!r} is "
        f"not recognised.  Expected {B04_PROTOCOL_NAME!r} or "
        f"{B04A_PROTOCOL_NAME!r}.  Rejecting before any artifact is "
        "written."
    )


def _run_validate_config_b04(
    *,
    config_path: Path,
    output_dir: Path,
    raw: dict[str, Any],
    config: MiniConfig,
    log_path: Path,
) -> int:
    """Validate-only path for the B04 protocol.

    Writes the historical B04 identity:

    * ``status.json`` -- ``task_id=TASK_ID``,
      ``config_version=MINI_VERSION``,
      ``registered_candidates=B04_CANDIDATE_NAMES``,
      ``model_parameter_cap=B04_MAX_PARAMETERS``.
    * ``resolved_config.json`` -- the resolved ``MiniConfig`` dict.
    * ``input_manifest_hashes.json`` -- config + A06 split SHA +
      registered candidates.
    * ``DONE.json`` -- the terminal mutual-exclusion file.
    * ``logs/run.log`` -- per-candidate parameter count + model
      version.

    This branch is the only path that references
    :data:`TASK_ID`, :data:`MINI_VERSION`,
    :data:`B04_CANDIDATE_NAMES`, and :data:`B04_MAX_PARAMETERS`.
    """

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID} config_path={config_path} protocol=B04")

    _write_json(output_dir / "resolved_config.json", config.as_dict())
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "note": "validate-only mode: no B01 freeze tables read",
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    # Verify the model registry can build both candidates (CPU) and
    # that the parameter count cap is respected.
    for cand in B04_CANDIDATE_NAMES:
        builder = get_model_builder(cand)
        model, _ = builder.factory(9, "cpu")
        count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if count > B04_MAX_PARAMETERS:
            raise MiniProtocolError(
                f"candidate {cand} has {count} parameters; exceeds B04 cap of "
                f"{B04_MAX_PARAMETERS}"
            )
        _log(f"candidate={cand} model_version={builder.version} parameters={count}")

    _write_json(
        output_dir / "status.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "protocol": B04_PROTOCOL_NAME,
            "status": "VALIDATED",
            "started_at_utc": _now_iso(),
            "ended_at_utc": _now_iso(),
            "mode": "validate-config",
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "model_parameter_cap": B04_MAX_PARAMETERS,
        },
    )
    write_status_files(
        output_dir,
        status="DONE",
        extra={
            "mode": "validate-config",
            "protocol": B04_PROTOCOL_NAME,
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
        },
    )
    return 0


def _run_validate_config_b04a(
    *,
    config_path: Path,
    output_dir: Path,
    raw: dict[str, Any],
    config: MiniConfig,
    log_path: Path,
) -> int:
    """Validate-only path for the B04A protocol.

    Writes the frozen B04A identity:

    * ``status.json`` -- ``task_id=B04A_TASK_ID``,
      ``config_version=B04A_CONFIG_VERSION``,
      ``protocol=B04A_PROTOCOL_NAME``,
      ``registered_candidates=B04A_ACTIVE_CANDIDATE_NAMES``,
      ``seeds=B04A_SEEDS``,
      ``deferred_candidates=[...from config...]``,
      ``model_parameter_cap=B04A_MAX_PARAMETERS``,
      ``feasibility_threshold=B04A_FEASIBILITY_THRESHOLD``.
    * ``resolved_config.json`` -- the resolved ``MiniConfig`` dict.
    * ``input_manifest_hashes.json`` -- config + A06 split SHA +
      B04A registered candidates.
    * ``DONE.json`` -- the terminal mutual-exclusion file.
    * ``logs/run.log`` -- per-(candidate, seed) parameter count.

    The B04A branch is the only path that writes
    :data:`B04A_TASK_ID`, :data:`B04A_CONFIG_VERSION`,
    :data:`B04A_ACTIVE_CANDIDATE_NAMES`,
    :data:`B04A_MAX_PARAMETERS`, and the B04A seeds.  It MUST NOT
    import :data:`TASK_ID`, :data:`MINI_VERSION`,
    :data:`B04_CANDIDATE_NAMES`, or :data:`B04_MAX_PARAMETERS` --
    identity leakage between protocols is forbidden.
    """

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(
        f"task_id={B04A_TASK_ID} config_path={config_path} protocol=B04A"
    )

    # Derive the deferred-candidate list from the config: every
    # entry with role="DEFERRED" is included so a Reviewer can
    # audit why SegFormer-B0 (or any future deferral) is not in
    # the registered active set.
    deferred_candidates: list[dict[str, Any]] = []
    for entry in raw.get("candidates", []):
        if str(entry.get("role", "")).upper() == "DEFERRED":
            deferred_candidates.append(
                {
                    "name": str(entry.get("name", "")),
                    "version": str(entry.get("version", entry.get("name", ""))),
                    "role": "DEFERRED",
                    "deferred_reason": str(
                        entry.get("deferred_reason", "")
                    ),
                }
            )

    _write_json(output_dir / "resolved_config.json", config.as_dict())
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "registered_candidates": list(B04A_ACTIVE_CANDIDATE_NAMES),
            "forbidden_candidates": list(B04A_FORBIDDEN_CANDIDATE_NAMES),
            "deferred_candidates": [c["name"] for c in deferred_candidates],
            "seeds": list(B04A_SEEDS),
            "note": "validate-only mode: no B01 freeze tables read",
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    # Verify the model registry can build the B04A active candidates
    # on CPU and that the parameter count cap (B04A_MAX_PARAMETERS)
    # is respected.  Exact parameter counts are checked too: B04A
    # binds the active candidates' parameter counts to the values
    # registered in B04A_EXACT_PARAMETER_COUNTS.
    for cand in B04A_ACTIVE_CANDIDATE_NAMES:
        builder = get_model_builder(cand)
        model, _ = builder.factory(9, "cpu")
        count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if count > B04A_MAX_PARAMETERS:
            raise MiniProtocolError(
                f"candidate {cand} has {count} parameters; exceeds B04A cap "
                f"of {B04A_MAX_PARAMETERS}"
            )
        expected_count = int(B04A_EXACT_PARAMETER_COUNTS.get(cand, count))
        if count != expected_count:
            raise MiniProtocolError(
                f"candidate {cand} has {count} parameters; B04A frozen "
                f"exact_parameter_count is {expected_count}"
            )
        _log(
            f"candidate={cand} model_version={builder.version} "
            f"parameters={count} seed_in_set=True"
        )

    for entry in deferred_candidates:
        _log(
            f"candidate={entry['name']} role=DEFERRED reason={entry['deferred_reason'][:80]}"
        )

    _write_json(
        output_dir / "status.json",
        {
            "task_id": B04A_TASK_ID,
            "config_version": B04A_CONFIG_VERSION,
            "protocol": B04A_PROTOCOL_NAME,
            "status": "VALIDATED",
            "started_at_utc": _now_iso(),
            "ended_at_utc": _now_iso(),
            "mode": "validate-config",
            "registered_candidates": list(B04A_ACTIVE_CANDIDATE_NAMES),
            "forbidden_candidates": list(B04A_FORBIDDEN_CANDIDATE_NAMES),
            "deferred_candidates": deferred_candidates,
            "seeds": list(B04A_SEEDS),
            "model_parameter_cap": B04A_MAX_PARAMETERS,
            "feasibility_threshold": B04A_FEASIBILITY_THRESHOLD,
        },
    )
    write_status_files(
        output_dir,
        status="DONE",
        extra={
            "mode": "validate-config",
            "protocol": B04A_PROTOCOL_NAME,
            "task_id": B04A_TASK_ID,
            "config_version": B04A_CONFIG_VERSION,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "registered_candidates": list(B04A_ACTIVE_CANDIDATE_NAMES),
            "seeds": list(B04A_SEEDS),
            "deferred_candidates": [c["name"] for c in deferred_candidates],
        },
    )
    return 0


# ---------------------------------------------------------------------------
# Synthetic CPU smoke
# ---------------------------------------------------------------------------


def _run_synthetic_cpu_smoke(
    config_path: Path,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
) -> int:
    """Run the B04 Mini end-to-end on synthetic data with CPU only."""

    output_dir = Path(output_dir).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID} mode=synthetic-cpu-smoke")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)

    # Build a synthetic-data override config (synthetic forces CPU).
    raw["training"]["device"] = "cpu"
    config = build_mini_config(
        raw,
        b01_freeze_dir="<SYNTHETIC>",
        data_root="<SYNTHETIC>",
        config_path=str(config_path),
    )

    device = resolve_device("cpu", allow_cpu_fallback=True)
    if str(device) != "cpu":
        raise MiniProtocolError("synthetic CPU smoke must run on cpu")

    # Test-only budget override: when B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS
    # is set, the synthetic smoke uses the supplied per-candidate wall
    # budget (in seconds) instead of the config's 45-minute value.  This
    # is exclusively a test hook to drive the STOPPED state in the
    # CLI integration tests.
    budget_override = os.environ.get("B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS")
    if budget_override is not None:
        try:
            override_seconds = float(budget_override)
        except ValueError:
            raise MiniProtocolError(
                f"B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS must parse as float; "
                f"got {budget_override!r}"
            )
        if override_seconds <= 0:
            raise MiniProtocolError(
                f"B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS must be > 0; "
                f"got {override_seconds}"
            )
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=float(override_seconds),
            max_wall_seconds_total=float(override_seconds) * 2,
            max_peak_cuda_mb=12288.0,
        )
        _log(
            f"test-only budget override: per_candidate={override_seconds}s "
            f"total={override_seconds * 2}s"
        )
    else:
        budget = resource_budget_from_config(
            {
                "max_wall_minutes_per_candidate": 45,
                "max_total_wall_minutes": 90,
                "max_peak_cuda_mb": 12288,
            }
        )

    _log(f"device={device}")

    # Build synthetic datasets.
    train_dataset, val_dataset, dataset_manifest, train_class_stats = build_synthetic_dataset(
        n_train_samples=int(SYNTHETIC_DEFAULTS["n_train_samples"]),
        n_val_samples=int(SYNTHETIC_DEFAULTS["n_val_samples"]),
        seed=int(SYNTHETIC_DEFAULTS["seed"]),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            "synthetic dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )
    _log(
        f"train_subjects={dataset_manifest['train_subjects']} "
        f"n_train={dataset_manifest['n_train_samples']} "
        f"n_val={dataset_manifest['n_val_samples']} "
        f"n_test={dataset_manifest['n_test_samples']}"
    )

    # Compute class weights from the synthetic TRAIN-only stats.
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
    _log(
        "class_weights="
        f"{[(c, round(class_weight_result.weights[c], 4)) for c in range(9)]}"
    )

    # Write the manifest and resolved config up front so the operator
    # can audit them even if a candidate later fails.
    _write_json(
        output_dir / "resolved_config.json",
        {**config.as_dict(), "mode": "synthetic-cpu-smoke"},
    )
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "synthetic": True,
            "synthetic_train_class_stats_sha256": _hash_dict(train_class_stats),
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    # Persist a small manifest so the reviewer can audit the run.
    _write_json(
        output_dir / "manifest.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "mode": "synthetic-cpu-smoke",
            "dataset_manifest": dataset_manifest,
            "train_class_stats": train_class_stats,
            "class_weight_summary": class_weight_result.as_dict(),
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "checkpoint_version": CHECKPOINT_VERSION,
            "b02_reference_val_fixed_iou": B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
            "b04_max_parameters": B04_MAX_PARAMETERS,
            "started_at_utc": _now_iso(),
        },
    )

    t_start = time.perf_counter()
    resume_from_per_candidate: dict[str, Path] | None = None
    if resume_from is not None:
        from topper_perception.neural.slp8_region_resume import (
            refuse_resume_for_done_run as _refuse,
        )
        _refuse(Path(resume_from))
        cfg_for_resume = build_mini_config(
            raw, b01_freeze_dir="<SYNTHETIC>", data_root="<SYNTHETIC>",
            config_path=str(config_path),
        )
        resume_from_per_candidate = _auto_detect_resume_candidates(
            Path(resume_from), cfg_for_resume
        )
        _log(
            f"resume_from={resume_from} -> {sorted(resume_from_per_candidate.keys())}"
        )
    result = run_mini(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "synthetic": True,
        },
        train_class_stats_source="synthetic_train_class_stats",
        synthetic=True,
        budget=budget,
        resume_from_per_candidate=resume_from_per_candidate,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)
    _log(
        f"completed in {result.wall_clock_seconds:.2f}s; "
        f"terminal_state={result.terminal_state} "
        f"overall_decision={result.overall_decision}; "
        f"feasible={result.n_candidates_feasible} "
        f"not_feasible={result.n_candidates_not_feasible} "
        f"failed={result.n_candidates_failed} "
        f"stopped={result.n_candidates_stopped}"
    )

    _write_json(
        output_dir / "manifest.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "mode": "synthetic-cpu-smoke",
            "dataset_manifest": dataset_manifest,
            "train_class_stats": train_class_stats,
            "class_weight_summary": class_weight_result.as_dict(),
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "checkpoint_version": CHECKPOINT_VERSION,
            "b02_reference_val_fixed_iou": B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
            "b04_max_parameters": B04_MAX_PARAMETERS,
            "started_at_utc": result.started_at_utc,
            "ended_at_utc": result.ended_at_utc,
            "wall_clock_seconds": result.wall_clock_seconds,
            "candidate_feasibility": {
                cand: {
                    "feasibility": cand_result.feasibility,
                    "reason": cand_result.reason,
                }
                for cand, cand_result in result.candidate_results.items()
            },
            "terminal_state": result.terminal_state,
            "overall_decision": result.overall_decision,
        },
    )

    # Terminal file follows ``result.terminal_state`` exactly:
    # DONE -> DONE.json + exit 0; FAILED -> FAILED.json + exit 1;
    # STOPPED -> STOPPED.json + exit 1.  ``write_status_files`` keeps
    # the three files mutually exclusive.
    write_status_files(
        output_dir,
        status=result.terminal_state,
        extra={
            "mode": "synthetic-cpu-smoke",
            "overall_decision": result.overall_decision,
            "wall_clock_seconds": result.wall_clock_seconds,
            "n_candidates_feasible": result.n_candidates_feasible,
            "n_candidates_not_feasible": result.n_candidates_not_feasible,
            "n_candidates_failed": result.n_candidates_failed,
            "n_candidates_stopped": result.n_candidates_stopped,
            "terminal_state": result.terminal_state,
        },
    )
    _write_json(output_dir / "status.json", {
        "task_id": TASK_ID,
        "config_version": MINI_VERSION,
        "status": result.terminal_state,
        "mode": "synthetic-cpu-smoke",
        "terminal_state": result.terminal_state,
        "started_at_utc": result.started_at_utc,
        "ended_at_utc": result.ended_at_utc,
        "wall_clock_seconds": result.wall_clock_seconds,
        "overall_decision": result.overall_decision,
        "n_candidates_feasible": result.n_candidates_feasible,
        "n_candidates_not_feasible": result.n_candidates_not_feasible,
        "n_candidates_failed": result.n_candidates_failed,
        "n_candidates_stopped": result.n_candidates_stopped,
    })
    # Terminal state machine:
    # DONE -> exit 0; FAILED / STOPPED -> exit 1.
    return 0 if result.terminal_state == "DONE" else 1


def _hash_dict(payload: Any) -> str:
    """Stable SHA-256 of a JSON payload."""

    import hashlib

    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Real B01 path (gated by --run-authorized)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Synthetic CPU smoke (B04A protocol)
# ---------------------------------------------------------------------------


def _run_synthetic_cpu_smoke_b04a(
    config_path: Path,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
    no_write: bool = False,
) -> int:
    """Run the B04A Mini on synthetic CPU data.

    The synthetic path keeps the B04A three-seed orchestration
    semantics but shrinks the per-candidate epoch budget so the
    smoke finishes in seconds, not minutes.  The protocol budget
    (45 min/candidate, 135 min total) is enforced by the B04A
    orchestrator's resource budget monitor; the smoke overrides the
    *real* B04A budget only via the test-only environment variable
    ``B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS`` so a
    reviewer's tiny-budget test can trigger STOPPED without rewiring
    the contract.

    The output policy is identical to the B04 smoke:
    ``--no-write`` exits 0 without writing any artifact; otherwise
    the existing output directory is refused (collision rule).
    """

    output_dir = Path(output_dir).resolve()
    check_output_dir_safety(output_dir)
    if not no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    if not no_write:
        log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        if no_write:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id=B04A protocol={B04A_PROTOCOL_NAME} config_version={B04A_CONFIG_VERSION}")
    _log(f"config_path={config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    if _protocol_of_config(raw) != B04A_PROTOCOL_NAME:
        raise MiniProtocolError(
            f"--synthetic-cpu-smoke-b04a requires protocol "
            f"{B04A_PROTOCOL_NAME!r}; got config_version "
            f"{raw.get('config_version')!r}"
        )

    # Validate the frozen protocol config first; then apply
    # synthetic-CPU overrides.  The B04A protocol mandates
    # ``device=cuda`` so the validator would otherwise reject the
    # ``cpu`` override.
    # Synthetic override: force CPU, shrink the per-candidate epoch budget.
    # The candidate set, seeds, threshold, and aggregator are unchanged.
    raw["training"]["device"] = "cpu"
    raw["training"]["max_epochs"] = int(SYNTHETIC_DEFAULTS_B04A["max_epochs_per_seed"])
    raw["training"]["min_epochs"] = int(SYNTHETIC_DEFAULTS_B04A["min_epochs"])
    raw["training"]["early_stopping"]["patience"] = int(
        SYNTHETIC_DEFAULTS_B04A["early_stopping_patience"]
    )
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

    device = resolve_device("cpu", allow_cpu_fallback=True)
    if str(device) != "cpu":
        raise MiniProtocolError("B04A synthetic CPU smoke must run on cpu")

    # Test-only budget override (seconds).  When set, the smoke uses
    # this value as the per-candidate wall budget; total = per * 3.
    budget_override = os.environ.get("B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS")
    if budget_override is not None:
        try:
            override_seconds = float(budget_override)
        except ValueError:
            raise MiniProtocolError(
                "B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS must "
                f"parse as float; got {budget_override!r}"
            )
        if override_seconds <= 0:
            raise MiniProtocolError(
                "B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS must "
                f"be > 0; got {override_seconds}"
            )
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=float(override_seconds),
            max_wall_seconds_total=float(override_seconds) * 3,
            max_peak_cuda_mb=8192.0,
        )
        _log(
            f"test-only budget override: per_candidate={override_seconds}s "
            f"total={override_seconds * 3}s"
        )
    else:
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=45 * 60.0,
            max_wall_seconds_total=135 * 60.0,
            max_peak_cuda_mb=8192.0,
        )

    _log(f"device={device}")
    _log(f"seeds={list(B04A_SEEDS)}")
    _log(f"active_candidates={list(B04A_ACTIVE_CANDIDATE_NAMES)}")

    # Build a small synthetic dataset (per-candidate shared TRAIN/VAL).
    train_dataset, val_dataset, dataset_manifest, train_class_stats = build_synthetic_dataset(
        n_train_samples=int(SYNTHETIC_DEFAULTS_B04A["n_train_samples"]),
        n_val_samples=int(SYNTHETIC_DEFAULTS_B04A["n_val_samples"]),
        seed=int(SYNTHETIC_DEFAULTS_B04A["seed"]),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            "B04A synthetic dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )
    _log(
        f"train_subjects={dataset_manifest['train_subjects']} "
        f"n_train={dataset_manifest['n_train_samples']} "
        f"n_val={dataset_manifest['n_val_samples']}"
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

    t_start = time.perf_counter()
    result = run_mini_b04a(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "synthetic": True,
        },
        train_class_stats_source="synthetic_train_class_stats",
        synthetic=True,
        budget=budget,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)

    _log(
        f"completed in {result.wall_clock_seconds:.4f}s; "
        f"terminal_state={result.terminal_state} "
        f"overall_decision={result.overall_decision} "
        f"advanced={list(result.advanced)} "
        f"near_tie_applied={result.near_tie_applied}"
    )

    if no_write:
        # Do NOT write any artifact; the runner integration smoke
        # script prints a single-line summary instead.
        n_advanced = len(result.advanced)
        n_feasible = int(result.n_candidates_feasible)
        n_failed = int(result.n_candidates_failed)
        n_stopped = int(result.n_candidates_stopped)
        all_seeds_passed = all(
            cand_result.n_seeds_failed == 0 and cand_result.n_seeds_stopped == 0
            and cand_result.feasibility in {"FEASIBLE", "INFEASIBLE"}
            for cand_result in result.candidate_results.values()
        )
        any_seed_feasible = any(
            seed_cand.feasibility == "FEASIBLE"
            for cand_result in result.candidate_results.values()
            for seed_cand in cand_result.per_seed.values()
        )
        print(
            f"B04A_SMOKE_NO_WRITE terminal_state={result.terminal_state} "
            f"feasible={n_feasible} failed={n_failed} stopped={n_stopped} "
            f"advanced={n_advanced} near_tie_applied={bool(result.near_tie_applied)} "
            f"all_seeds_attempted={bool(all_seeds_passed)} "
            f"any_seed_feasible={bool(any_seed_feasible)}"
        )
        return 0

    config_sha256 = file_sha256(config_path)
    _write_b04a_run_bundle(
        output_dir=output_dir,
        result=result,
        config_sha256=config_sha256,
    )

    # Terminal file (DONE/FAILED/STOPPED) — mutually exclusive.
    write_status_files(
        output_dir,
        status=result.terminal_state,
        extra={
            "task_id": result.config.task_id,
            "protocol": result.config.protocol,
            "config_version": result.config.config_version,
            "mode": "synthetic-cpu-smoke-b04a",
            "overall_decision": result.overall_decision,
            "advanced": list(result.advanced),
            "near_tie_applied": bool(result.near_tie_applied),
            "wall_clock_seconds": result.wall_clock_seconds,
            "n_candidates_feasible": int(result.n_candidates_feasible),
            "n_candidates_not_feasible": int(result.n_candidates_not_feasible),
            "n_candidates_failed": int(result.n_candidates_failed),
            "n_candidates_stopped": int(result.n_candidates_stopped),
            "terminal_state": result.terminal_state,
        },
    )
    return 0 if result.terminal_state == "DONE" else 1


def _load_b01_freeze_and_contract(
    raw: dict[str, Any],
    b01_freeze_dir: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Load B01 freeze tables and run the B01 input contract.

    Returns a dict with:
      * ``freeze`` -- the B01 freeze handle
      * ``snapshot`` -- the :class:`B01FreezeSnapshot`
      * ``b01_contract_report`` -- dict report
      * ``freeze_manifest_path`` / ``train_class_stats_path``
      * ``fm_file_sha`` / ``train_class_stats_sha256``
      * ``train_class_stats`` -- parsed JSON
      * ``b01_expected`` -- the :class:`B01ContractExpected`

    The real B01 input contract is enforced **before** any training
    artifact is written:

    1. ``freeze_manifest.json`` and ``train_class_stats.json`` must
       exist on disk.
    2. The on-disk ``freeze_manifest.json`` SHA must match the SHA
       recorded in the freeze handle (via
       :func:`check_freeze_manifest_file_consistency`).
    3. A :class:`B01FreezeSnapshot` is constructed and run through
       :func:`verify_b01_contract` which fail-closes on
       train/val/test count, subject count, A06 split SHA, provenance,
       source review status, setting, cover, or freeze manifest SHA
       mismatch.  The snapshot is built **from the freeze data** --
       no constant defaults may be substituted.

    The B01 freeze tables DO include the held-out TEST 495 rows in
    their ``test_manifest.csv``.  That is a *structural* fact of the
    freeze.  However, both protocols insist the loaded dataset
    reports ``n_test_samples=0`` and TEST labels / onehots are
    never reachable from the runner.  We explicitly do not pass
    TEST rows to the dataset builder:
    ``load_b01_freeze_tables(..., load_test=False)`` returns a
    freeze whose ``_test_rows`` is ``None``.
    """

    if not b01_freeze_dir.is_dir():
        raise FileNotFoundError(
            f"B01 freeze directory not found: {b01_freeze_dir}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    freeze = load_b01_freeze_tables(b01_freeze_dir, load_test=False)
    if freeze._test_rows is not None:  # noqa: SLF001
        raise MiniProtocolError(
            "TEST rows were loaded by load_b01_freeze_tables(..., load_test=False); "
            "this must not happen"
        )

    n_train = len(freeze.train_rows)
    n_val = len(freeze.val_rows)
    if n_train == 0 or n_val == 0:
        raise MiniProtocolError(
            f"B01 freeze has zero rows for an essential split: "
            f"train={n_train}, val={n_val}"
        )

    freeze_manifest_path = b01_freeze_dir / "freeze_manifest.json"
    train_class_stats_path = b01_freeze_dir / "train_class_stats.json"
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(
            f"freeze_manifest.json missing in {b01_freeze_dir}"
        )
    if not train_class_stats_path.is_file():
        raise FileNotFoundError(
            f"train_class_stats.json missing in {b01_freeze_dir}"
        )
    train_class_stats = json.loads(
        train_class_stats_path.read_text(encoding="utf-8")
    )

    b01_expected = build_b01_contract_expected(raw)
    fm_file_sha = sha256_file(freeze_manifest_path)
    check_freeze_manifest_file_consistency(
        b01_freeze_dir,
        freeze_manifest_sha256=b01_expected.freeze_manifest_core_sha256,
    )
    snapshot = B01FreezeSnapshot.from_freeze_tables(
        freeze_dir=b01_freeze_dir,
        train_rows=freeze.train_rows,
        val_rows=freeze.val_rows,
        test_rows=None,  # explicitly None -- TEST is never loaded
        freeze_manifest=freeze.freeze_manifest,
    )
    b01_contract_report = verify_b01_contract(
        snapshot, b01_expected
    ).as_dict()
    return {
        "freeze": freeze,
        "snapshot": snapshot,
        "b01_contract_report": b01_contract_report,
        "freeze_manifest_path": freeze_manifest_path,
        "train_class_stats_path": train_class_stats_path,
        "fm_file_sha": fm_file_sha,
        "train_class_stats_sha256": sha256_file(train_class_stats_path),
        "train_class_stats": train_class_stats,
        "b01_expected": b01_expected,
    }


def _auto_detect_resume_candidates_b04a(
    resume_from: Path,
    config: "MiniConfig",
) -> dict[str, dict[int, Path]]:
    """Auto-detect per-(candidate, seed) ``last.pt`` files for B04A.

    The CLI takes a single ``--resume-from`` path that is the
    output directory of a previous (interrupted) B04A run.  This
    function walks the directory and returns a mapping
    ``{candidate_name: {seed: last_pt_path}}`` for
    (candidate, seed) pairs that completed at least one epoch.
    B04A never advances without a per-seed budget, so the
    resume contract is per-seed.

    Same DONE / unknown-state / empty-state refuse rules as the
    B04 helper.
    """

    from topper_perception.neural.slp8_region_resume import ResumeRefusedError
    from topper_perception.neural.slp8_region_mini import (
        MiniProtocolError,
        refuse_resume_for_done_run,
    )

    if not resume_from.is_dir():
        raise MiniProtocolError(
            f"--resume-from path is not a directory: {resume_from}"
        )
    refuse_resume_for_done_run(resume_from)
    status_path = resume_from / "status.json"
    if not status_path.is_file():
        raise MiniProtocolError(
            f"--resume-from source lacks status.json: {resume_from}"
        )
    try:
        source_status = json.loads(status_path.read_text(encoding="utf-8"))
        source_terminal_state = str(
            source_status.get("terminal_state", source_status.get("status", ""))
        )
    except Exception as exc:
        raise MiniProtocolError(
            f"--resume-from source status.json is unreadable: {exc}"
        ) from exc
    if source_terminal_state not in {"RUNNING", "FAILED", "STOPPED"}:
        raise MiniProtocolError(
            "--resume-from source must have terminal_state RUNNING, FAILED, "
            f"or STOPPED; got {source_terminal_state!r}"
        )
    result: dict[str, dict[int, Path]] = {}
    for cand in config.candidates:
        for seed in config.seeds:
            last_pt = (
                resume_from
                / "checkpoints"
                / cand
                / f"seed_{seed:04d}"
                / "last.pt"
            )
            if last_pt.is_file():
                result.setdefault(cand, {})[int(seed)] = last_pt
    if not result:
        raise MiniProtocolError(
            "--resume-from source contains no completed-epoch (candidate, "
            "seed) last.pt; there is no safe state to resume"
        )
    return result


def _run_real_b01_b04(
    config_path: Path,
    output_dir: Path,
    b01_freeze_dir: Path,
    dataset_root: Path,
    *,
    resume_from: Path | None,
    config: "MiniConfig",
    b01: dict[str, Any],
    log_path: Path,
) -> int:
    """Run the B04 (single-seed) real B01 path.

    Pulled out of ``_run_real_b01`` so the protocol dispatch is
    trivially testable and the B04A real path can be implemented
    symmetrically without falling through the wrong orchestrator.
    """

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID} mode=real-b01 protocol={B04_PROTOCOL_NAME}")

    device = resolve_device("cuda", allow_cpu_fallback=False)
    _log(f"device={device}")

    train_subjects = sorted(
        {row.subject_id for row in b01["freeze"].train_rows}
    )
    val_subjects = sorted(
        {row.subject_id for row in b01["freeze"].val_rows}
    )
    if not verify_subject_isolation(train_subjects, val_subjects):
        raise MiniProtocolError(
            "TRAIN/VAL subject overlap detected in B01 freeze"
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    class_weight_result = compute_class_weights(b01["train_class_stats"])
    assert_class_weight_invariants(class_weight_result)

    from topper_perception.neural.slp8_region_dataset import (
        build_smoke_dataset,
    )
    train_dataset, val_dataset, dataset_manifest = build_smoke_dataset(
        b01_freeze_dir=b01_freeze_dir,
        dataset_root=dataset_root,
        seed=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("seed", 42)
        ),
        n_train_subjects=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("n_train_subjects", len(train_subjects))
        ),
        n_val_subjects=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("n_val_subjects", len(val_subjects))
        ),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            f"real B01 dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )

    _write_json(
        output_dir / "resolved_config.json", config.as_dict()
    )
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_sha256": file_sha256(config_path),
            "freeze_manifest_file_sha256": b01["fm_file_sha"],
            "freeze_manifest_core_sha256": (
                b01["b01_expected"].freeze_manifest_core_sha256
            ),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "b01_freeze_dir": REDACTED_LOCAL_PATH,
            "dataset_root": REDACTED_LOCAL_PATH,
            "train_class_stats_sha256": b01["train_class_stats_sha256"],
            "b01_contract_report": b01["b01_contract_report"],
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    resume_from_per_candidate: dict[str, Path] | None = None
    if resume_from is not None:
        resume_from_per_candidate = _auto_detect_resume_candidates(
            Path(resume_from), config
        )
        _log(
            f"resume_from={resume_from} -> "
            f"{sorted(resume_from_per_candidate.keys())}"
        )

    t_start = time.perf_counter()
    result = run_mini(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": file_sha256(config_path),
            "freeze_manifest_sha256": (
                b01["b01_expected"].freeze_manifest_core_sha256
            ),
            "freeze_manifest_file_sha256": b01["fm_file_sha"],
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "train_class_stats_sha256": b01["train_class_stats_sha256"],
        },
        train_class_stats_source="b01_train_class_stats.json",
        synthetic=False,
        b01_contract_report=b01["b01_contract_report"],
        resume_from_per_candidate=resume_from_per_candidate,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)
    _log(
        f"completed in {result.wall_clock_seconds:.2f}s; "
        f"terminal_state={result.terminal_state} "
        f"overall_decision={result.overall_decision}"
    )

    write_status_files(
        output_dir,
        status=result.terminal_state,
        extra={
            "task_id": TASK_ID,
            "protocol": config.protocol,
            "config_version": config.config_version,
            "mode": "real-b01",
            "overall_decision": result.overall_decision,
            "wall_clock_seconds": result.wall_clock_seconds,
            "n_candidates_feasible": result.n_candidates_feasible,
            "n_candidates_not_feasible": result.n_candidates_not_feasible,
            "n_candidates_failed": result.n_candidates_failed,
            "n_candidates_stopped": result.n_candidates_stopped,
            "terminal_state": result.terminal_state,
        },
    )
    _write_json(
        output_dir / "status.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "protocol": B04_PROTOCOL_NAME,
            "status": result.terminal_state,
            "mode": "real-b01",
            "terminal_state": result.terminal_state,
            "overall_decision": result.overall_decision,
            "wall_clock_seconds": result.wall_clock_seconds,
            "n_candidates_feasible": result.n_candidates_feasible,
            "n_candidates_not_feasible": result.n_candidates_not_feasible,
            "n_candidates_failed": result.n_candidates_failed,
            "n_candidates_stopped": result.n_candidates_stopped,
        },
    )
    return 0 if result.terminal_state == "DONE" else 1


def _run_real_b01_b04a(
    config_path: Path,
    output_dir: Path,
    b01_freeze_dir: Path,
    dataset_root: Path,
    *,
    resume_from: Path | None,
    config: "MiniConfig",
    b01: dict[str, Any],
    log_path: Path,
) -> int:
    """Run the B04A (three-seed) real B01 path.

    Pulled out of ``_run_real_b01`` so the protocol dispatch is
    trivially testable.  The B04A real path:

    1. Verifies the B01 contract (shared with B04).
    2. Builds the real B01 dataset (shared with B04).
    3. Calls :func:`run_mini_b04a`, which iterates
       ``candidates × seeds`` with per-seed identity, per-seed
       budget accumulator, ``all_seeds_must_succeed``, and the
       B04A candidate-level decision rules.
    4. Auto-detects per-(candidate, seed) resume mapping from
       ``checkpoints/<candidate>/seed_<seed>/last.pt``.
    5. Writes B04A artifacts (manifest, candidate_decision with
       tiebreak records, budget_report, per-seed identity
       siblings) via :func:`_write_b04a_run_bundle` and the
       mutually-exclusive terminal file.

    No real data is loaded by this task; the dispatch is
    unit-tested via mocks/stubs.  ``--run-authorized`` is still
    required (B04 and B04A both gate on it).
    """

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(
        f"task_id={config.task_id} mode=real-b01 protocol={B04A_PROTOCOL_NAME}"
    )
    _log(f"seeds={list(config.seeds)}")
    _log(f"active_candidates={list(config.candidates)}")

    device = resolve_device("cuda", allow_cpu_fallback=False)
    _log(f"device={device}")

    train_subjects = sorted(
        {row.subject_id for row in b01["freeze"].train_rows}
    )
    val_subjects = sorted(
        {row.subject_id for row in b01["freeze"].val_rows}
    )
    if not verify_subject_isolation(train_subjects, val_subjects):
        raise MiniProtocolError(
            "TRAIN/VAL subject overlap detected in B01 freeze"
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    class_weight_result = compute_class_weights(b01["train_class_stats"])
    assert_class_weight_invariants(class_weight_result)

    from topper_perception.neural.slp8_region_dataset import (
        build_smoke_dataset,
    )
    train_dataset, val_dataset, dataset_manifest = build_smoke_dataset(
        b01_freeze_dir=b01_freeze_dir,
        dataset_root=dataset_root,
        seed=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("seed", 42)
        ),
        n_train_subjects=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("n_train_subjects", len(train_subjects))
        ),
        n_val_subjects=int(
            raw.get("dataset", {})
            .get("smoke_subset", {})
            .get("n_val_subjects", len(val_subjects))
        ),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            f"real B01 dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )

    budget = ResourceBudget(
        max_wall_seconds_per_candidate=45 * 60.0,
        max_wall_seconds_total=135 * 60.0,
        max_peak_cuda_mb=8192.0,
    )

    resume_per_candidate_seed: dict[str, dict[int, Path]] | None = None
    if resume_from is not None:
        resume_per_candidate_seed = (
            _auto_detect_resume_candidates_b04a(
                Path(resume_from), config
            )
        )
        _log(
            f"resume_from={resume_from} -> "
            + str(
                {
                    cand: sorted(seeds.keys())
                    for cand, seeds in sorted(
                        resume_per_candidate_seed.items()
                    )
                }
            )
        )

    config_sha256 = file_sha256(config_path)
    _write_json(
        output_dir / "resolved_config.json", config.as_dict()
    )
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_sha256": config_sha256,
            "freeze_manifest_file_sha256": b01["fm_file_sha"],
            "freeze_manifest_core_sha256": (
                b01["b01_expected"].freeze_manifest_core_sha256
            ),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "b01_freeze_dir": REDACTED_LOCAL_PATH,
            "dataset_root": REDACTED_LOCAL_PATH,
            "train_class_stats_sha256": b01["train_class_stats_sha256"],
            "b01_contract_report": b01["b01_contract_report"],
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    t_start = time.perf_counter()
    result = run_mini_b04a(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": config_sha256,
            "freeze_manifest_sha256": (
                b01["b01_expected"].freeze_manifest_core_sha256
            ),
            "freeze_manifest_file_sha256": b01["fm_file_sha"],
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "train_class_stats_sha256": b01["train_class_stats_sha256"],
        },
        train_class_stats_source="b01_train_class_stats.json",
        synthetic=False,
        budget=budget,
        b01_contract_report=b01["b01_contract_report"],
        resume_from_per_candidate_seed=resume_per_candidate_seed,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)
    _log(
        f"completed in {result.wall_clock_seconds:.4f}s; "
        f"terminal_state={result.terminal_state} "
        f"overall_decision={result.overall_decision} "
        f"advanced={list(result.advanced)} "
        f"near_tie_applied={bool(result.near_tie_applied)}"
    )

    _write_b04a_run_bundle(
        output_dir=output_dir,
        result=result,
        config_sha256=config_sha256,
    )
    write_status_files(
        output_dir,
        status=result.terminal_state,
        extra={
            "task_id": config.task_id,
            "protocol": config.protocol,
            "config_version": config.config_version,
            "mode": "real-b01-b04a",
            "overall_decision": result.overall_decision,
            "advanced": list(result.advanced),
            "near_tie_applied": bool(result.near_tie_applied),
            "wall_clock_seconds": result.wall_clock_seconds,
            "n_candidates_feasible": result.n_candidates_feasible,
            "n_candidates_not_feasible": result.n_candidates_not_feasible,
            "n_candidates_failed": result.n_candidates_failed,
            "n_candidates_stopped": result.n_candidates_stopped,
            "terminal_state": result.terminal_state,
        },
    )
    return 0 if result.terminal_state == "DONE" else 1


def _run_real_b01(
    config_path: Path,
    output_dir: Path,
    b01_freeze_dir: Path,
    dataset_root: Path,
    *,
    resume_from: Path | None = None,
) -> int:
    """Run the B04 / B04A Mini on real B01 freeze tables.  Requires ``--run-authorized``.

    This function is the protocol-dispatch entry point for the
    real B01 path.  After validating the config and loading the
    B01 freeze handle, it dispatches to:

    * :func:`_run_real_b01_b04` for ``config.protocol == "B04"``;
    * :func:`_run_real_b01_b04a` for ``config.protocol == "B04A"``.

    The cross-protocol path is fail-closed: an unknown protocol
    raises :class:`MiniProtocolError` BEFORE any training
    artifact is written.  The shared B01 contract loading lives
    in :func:`_load_b01_freeze_and_contract` and runs for both
    protocols, so a contract failure is auditable on either
    path.

    No real data is loaded by the current task; the dispatch
    is exercised by ``test_b04a_runner_integration.py`` via
    ``monkeypatch``-replaced ``run_mini`` / ``run_mini_b04a``.
    """

    output_dir = Path(output_dir).resolve()
    b01_freeze_dir = Path(b01_freeze_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    config = build_mini_config(
        raw,
        b01_freeze_dir=str(b01_freeze_dir),
        data_root=str(dataset_root),
        config_path=str(config_path),
    )

    if config.protocol == B04_PROTOCOL_NAME:
        _log_path = log_path
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{_now_iso()}] task_id={TASK_ID} config_path={config_path} "
                f"protocol={B04_PROTOCOL_NAME} mode=real-b01-dispatch\n"
            )
        b01 = _load_b01_freeze_and_contract(
            raw, b01_freeze_dir, dataset_root
        )
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{_now_iso()}] B01 contract verified: "
                f"train={b01['snapshot'].train_count} "
                f"val={b01['snapshot'].val_count} "
                f"test={b01['snapshot'].structural_test.sample_count} "
                f"a06={b01['snapshot'].a06_split_sha256[:12]}…\n"
            )
        return _run_real_b01_b04(
            config_path=config_path,
            output_dir=output_dir,
            b01_freeze_dir=b01_freeze_dir,
            dataset_root=dataset_root,
            resume_from=resume_from,
            config=config,
            b01=b01,
            log_path=_log_path,
        )

    if config.protocol == B04A_PROTOCOL_NAME:
        _log_path = log_path
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{_now_iso()}] task_id={config.task_id} config_path={config_path} "
                f"protocol={B04A_PROTOCOL_NAME} mode=real-b01-dispatch\n"
            )
        b01 = _load_b01_freeze_and_contract(
            raw, b01_freeze_dir, dataset_root
        )
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{_now_iso()}] B01 contract verified: "
                f"train={b01['snapshot'].train_count} "
                f"val={b01['snapshot'].val_count} "
                f"test={b01['snapshot'].structural_test.sample_count} "
                f"a06={b01['snapshot'].a06_split_sha256[:12]}…\n"
            )
        return _run_real_b01_b04a(
            config_path=config_path,
            output_dir=output_dir,
            b01_freeze_dir=b01_freeze_dir,
            dataset_root=dataset_root,
            resume_from=resume_from,
            config=config,
            b01=b01,
            log_path=_log_path,
        )

    raise MiniProtocolError(
        f"_run_real_b01: config.protocol={config.protocol!r} is not "
        f"recognised.  Expected {B04_PROTOCOL_NAME!r} or "
        f"{B04A_PROTOCOL_NAME!r}.  Rejecting before any training "
        "artifact is written."
    )


def _auto_detect_resume_candidates(
    resume_from: Path,
    config: "MiniConfig",
) -> dict[str, Path]:
    """Auto-detect per-candidate ``last.pt`` files under ``resume_from``.

    The CLI takes a single ``--resume-from`` path that is the output
    directory of a previous (interrupted) B04 run.  This function
    walks the directory and returns a mapping
    ``{candidate_name: last_pt_path}`` for candidates that completed at
    least one epoch.  A serial run may be interrupted while a later
    candidate has not started; that candidate must start fresh instead
    of making the whole experiment non-resumable.  The function refuses
    DONE and unknown terminal states, and refuses a source with no
    completed-epoch checkpoint at all.
    """

    from topper_perception.neural.slp8_region_resume import ResumeRefusedError
    from topper_perception.neural.slp8_region_mini import (
        MiniProtocolError,
        refuse_resume_for_done_run,
    )

    if not resume_from.is_dir():
        raise MiniProtocolError(
            f"--resume-from path is not a directory: {resume_from}"
        )
    refuse_resume_for_done_run(resume_from)
    status_path = resume_from / "status.json"
    if not status_path.is_file():
        raise MiniProtocolError(
            f"--resume-from source lacks status.json: {resume_from}"
        )
    try:
        source_status = json.loads(status_path.read_text(encoding="utf-8"))
        source_terminal_state = str(
            source_status.get("terminal_state", source_status.get("status", ""))
        )
    except Exception as exc:
        raise MiniProtocolError(
            f"--resume-from source status.json is unreadable: {exc}"
        ) from exc
    if source_terminal_state not in {"RUNNING", "FAILED", "STOPPED"}:
        raise MiniProtocolError(
            "--resume-from source must have terminal_state RUNNING, FAILED, "
            f"or STOPPED; got {source_terminal_state!r}"
        )
    result: dict[str, Path] = {}
    for cand in config.candidates:
        last_pt = resume_from / "checkpoints" / cand / "last.pt"
        if last_pt.is_file():
            result[cand] = last_pt
    if not result:
        raise MiniProtocolError(
            "--resume-from source contains no completed-epoch last.pt; "
            "there is no safe state to resume"
        )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the B04 Mini config JSON.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for run artefacts.",
    )
    parser.add_argument(
        "--validate-config", dest="validate_config",
        action="store_true",
        help="Only validate the config; do not run anything.",
    )
    parser.add_argument(
        "--synthetic-cpu-smoke", dest="synthetic_cpu_smoke",
        action="store_true",
        help="Run the B04 Mini on synthetic CPU data (no real B01).",
    )
    parser.add_argument(
        "--synthetic-cpu-smoke-b04a", dest="synthetic_cpu_smoke_b04a",
        action="store_true",
        help=(
            "Run the B04A Mini on synthetic CPU data (3 seeds, 3 "
            "candidates, smaller epoch/sample budget).  Refuses any "
            "non-B04A config.  This is the runner-integration smoke "
            "for the B04A architecture expansion protocol."
        ),
    )
    parser.add_argument(
        "--no-write", dest="no_write", action="store_true",
        help=(
            "Run the synthetic smoke end-to-end but do NOT write any "
            "artifact.  The CLI exits 0 and prints a single-line "
            "summary.  This is the no-write variant used by the "
            "runner-integration smoke script and by Codex Reviewer "
            "audit runs."
        ),
    )
    parser.add_argument(
        "--b01-freeze-dir", type=Path, default=None,
        help="Path to the B01 freeze directory (real run only).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Path to the SLP8 dataset root (real run only).",
    )
    parser.add_argument(
        "--run-authorized", dest="run_authorized",
        action="store_true",
        help="REQUIRED for any real B01 run; the B04 protocol task does not set this.",
    )
    parser.add_argument(
        "--resume-from", dest="resume_from", type=Path, default=None,
        help=(
            "Path to a previous (interrupted) B04 output directory. "
            "Refuses to resume a DONE run; the resume otherwise auto-detects "
            "checkpoints/<candidate>/last.pt and continues training."
        ),
    )

    args = parser.parse_args(argv)

    try:
        # Mutual-exclusion: --validate-config and --synthetic-cpu-smoke
        # and a real run cannot coexist.
        if (
            args.validate_config
            + args.synthetic_cpu_smoke
            + args.synthetic_cpu_smoke_b04a
            > 1
        ):
            raise MiniProtocolError(
                "--validate-config, --synthetic-cpu-smoke and "
                "--synthetic-cpu-smoke-b04a are mutually exclusive"
            )

        if args.validate_config:
            return _run_validate_config(args.config, args.output_dir)

        if args.synthetic_cpu_smoke:
            return _run_synthetic_cpu_smoke(
                args.config, args.output_dir, resume_from=args.resume_from
            )

        if args.synthetic_cpu_smoke_b04a:
            return _run_synthetic_cpu_smoke_b04a(
                args.config,
                args.output_dir,
                resume_from=args.resume_from,
                no_write=bool(args.no_write),
            )

        if args.b01_freeze_dir is not None or args.dataset_root is not None:
            if not args.run_authorized:
                raise MiniProtocolError(
                    "B01 freeze or dataset-root paths were supplied but "
                    "--run-authorized was NOT set.  B04 forbids the real B01 "
                    "path without explicit owner authorization.  Re-run with "
                    "--run-authorized to proceed."
                )
            if args.b01_freeze_dir is None or args.dataset_root is None:
                raise MiniProtocolError(
                    "real B01 run requires both --b01-freeze-dir and --dataset-root"
                )
            return _run_real_b01(
                args.config, args.output_dir, args.b01_freeze_dir, args.dataset_root,
                resume_from=args.resume_from,
            )

        # No explicit mode supplied — default to validate-config.
        return _run_validate_config(args.config, args.output_dir)
    except Exception as exc:
        # When the exception is an ``OutputCollisionError`` or an
        # authorization-rejection ``MiniProtocolError`` we MUST NOT
        # create any file in the output directory: doing so would
        # silently mutate the directory the operator just asked us
        # to leave alone.  Only other (post-validation) errors write
        # ``FAILED.json`` / ``status.json`` so the directory is
        # auditable.
        from topper_perception.neural.slp8_region_mini import (
            OutputCollisionError,
        )

        non_mutating = (
            isinstance(exc, OutputCollisionError)
            or "--run-authorized was NOT set" in str(exc)
            or "real B01 run requires both" in str(exc)
        )
        if non_mutating:
            print(f"REJECTED: {exc}", file=sys.stderr)
            return 2

        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failed = {
            "status": "FAILED",
            "task_id": TASK_ID,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "ended_at_utc": _now_iso(),
        }
        try:
            write_status_files(
                output_dir,
                status="FAILED",
                extra={
                    "mode": "validate-config"
                    if args.validate_config
                    else (
                        "synthetic-cpu-smoke"
                        if args.synthetic_cpu_smoke
                        else "real-b01"
                    ),
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        _write_json(output_dir / "status.json", {
            "task_id": TASK_ID,
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
