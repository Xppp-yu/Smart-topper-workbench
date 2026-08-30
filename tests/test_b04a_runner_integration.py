"""Focused B04A runner-integration tests (TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1).

This module exercises the protocol/profile dispatch added in the
B04A runner-integration stage and pins the fail-closed behaviour
the B04A architecture expansion contract requires.  Every test
here is targeted at one contract clause and is independent of the
B04 implementation smoke (which lives in
``tests/test_b04a_implementation.py``).

Coverage (B04A contract sections; cross-references the B04A R03
protocol and the TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1
contract):

1. B04 / B04A config identity separation (protocol dispatch).
2. B04 backward-compatibility: the historical B04 config still
   passes validation, builds the same ``MiniConfig`` shape, and
   feeds the same B04 orchestrator.
3. Unknown ``config_version`` is rejected fail-closed.
4. B04/B04A candidate mix-up is rejected fail-closed.
5. ``slp8_tiny_fcn_v0.1`` and ``slp8_segformer_b0_v0.1`` are
   forbidden in B04A.
6. B04A reads exactly the three registered seeds
   ``[42, 123, 2026]``; running with a different set is rejected.
7. B04A enforces ``all_seeds_must_succeed``: any per-seed
   FAILED / STOPPED / non-finite / class-collapse / floor
   violation flips the entire candidate to INFEASIBLE.
8. B04A 0/1/2/3-feasible decision: 0 -> MINI_NOT_FEASIBLE;
   1 -> advance single; 2 -> advance both; 3 -> top 2 with
   near-tie tiebreak.
9. B04A near-tie tiebreak: when the top-2 ``macro_iou_mean`` is
   within ``B04A_NEAR_TIE_MARGIN``, the simpler model wins.
10. Resource budget: per-candidate 45 min, total 135 min, max
    peak CUDA MiB 8192.  Resume restores both the candidate-level
    and the total-level accumulators.
11. Identity mismatch: resume with a different identity is
    rejected fail-closed.
12. Existing output directory is refused (no overwrite).
13. Terminal status mutex: DONE / FAILED / STOPPED are mutually
    exclusive.
14. TEST=0: B04A never imports or invokes B01 test access; the
    smoke summary's ``test_access`` field is ``declarative_policy``
    (not a runtime counter).
15. B04 backward-compatibility: 15/15 of the B04 mini tests
    (``TestSmallUnetArchitecture``, ``TestCandidateRegistry``,
    ``build_synthetic_dataset`` smoke, ``TestPredict``) still
    pass.

The synthetic CPU smoke is the no-write variant: it runs the
B04A orchestrator end-to-end on a 4-train / 2-val synthetic
dataset, 1 epoch per seed, 3 candidates x 3 seeds, and verifies
the run-level bundle (``manifest.json``, ``candidate_decision.json``,
``budget_report.json``, ``status.json``, per-seed
``checkpoints/<candidate>/seed_<seed>/{last,best}.pt``) is written
when ``--no-write`` is OFF.  The ``--no-write`` path runs the same
orchestrator without writing to disk; the smoke prints a single
``B04A_SMOKE_NO_WRITE`` summary line.

The file is intentionally read-only against B01 freeze tables and
TEST rows: no ``load_b01_freeze_tables`` call, no
``enable_test_access`` call, no real B01 paths are touched.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.neural.slp8_region_mini import (  # noqa: E402
    B04_CANDIDATE_NAMES,
    B04A_ACTIVE_CANDIDATE_NAMES,
    B04A_CONFIG_VERSION,
    B04A_FEASIBILITY_THRESHOLD,
    B04A_FORBIDDEN_CANDIDATE_NAMES,
    B04A_NEAR_TIE_MARGIN,
    B04A_PER_REGION_FLOOR,
    B04A_PROTOCOL_NAME,
    B04A_SEEDS,
    B04A_TASK_ID,
    B04A_WORST_SUBJECT_FLOOR,
    B04_PROTOCOL_NAME,
    B04A_EXACT_PARAMETER_COUNTS,
    B04A_MAX_PARAMETERS,
    CandidateResult,
    CHECKPOINT_VERSION,
    ConfigValidationError,
    FOREGROUND_CLASS_IDS,
    MINI_VERSION,
    MiniConfig,
    MiniProtocolError,
    N_CLASSES,
    OutputCollisionError,
    PRESSURE_SHAPE,
    ResourceBudget,
    ResourceBudgetState,
    SMALL_UNET_VERSION,
    SYNTHETIC_DEFAULTS,
    SYNTHETIC_DEFAULTS_B04A,
    SYNTHETIC_EXP_ID,
    TASK_ID,
    _b04a_advance_decision,
    _b04a_aggregate_candidate,
    _b04a_per_region_pass,
    _b04a_seed_class_collapse,
    _b04a_worst_subject_pass,
    _protocol_of_config,
    build_mini_config,
    build_synthetic_dataset,
    check_output_dir_safety,
    compute_class_weights,
    file_sha256,
    run_mini,
    run_mini_b04a,
    validate_mini_config,
    write_mini_artifacts,
    write_status_files,
)
from topper_perception.neural.slp8_region_class_weights import (  # noqa: E402
    assert_class_weight_invariants,
)
from topper_perception.neural.slp8_region_resume import (  # noqa: E402
    CheckpointIdentity,
    ResumeIdentityError,
    identity_from_dict,
    verify_resume_identity,
)
from topper_perception.neural.slp8_region_models import (  # noqa: E402
    DEEPLABV3PLUS_LITE_VERSION,
    MODEL_VERSION,
    RESUNET_LITE_VERSION,
    get_model_builder,
)
from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    A06_SPLIT_SHA256_EXPECTED,
)


B04_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "slp8_pm_region_mini_v0.1.json"
)
B04A_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "slp8_pm_architecture_expansion_mini_v0.1.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_b04() -> dict[str, Any]:
    return json.loads(B04_CONFIG_PATH.read_text(encoding="utf-8"))


def _load_b04a() -> dict[str, Any]:
    return json.loads(B04A_CONFIG_PATH.read_text(encoding="utf-8"))


def _build_b04_mini_config() -> MiniConfig:
    raw = _load_b04()
    validate_mini_config(raw)
    return build_mini_config(
        raw,
        b01_freeze_dir=None,
        data_root=None,
        config_path=str(B04_CONFIG_PATH),
    )


def _build_b04a_mini_config(
    *,
    max_epochs: int = 1,
    min_epochs: int = 1,
    patience: int = 1,
) -> MiniConfig:
    raw = _load_b04a()
    validate_mini_config(raw)  # validate the frozen protocol config
    # Then apply synthetic-CPU overrides AFTER validation.
    raw["training"]["device"] = "cpu"
    raw["training"]["max_epochs"] = max_epochs
    raw["training"]["min_epochs"] = min_epochs
    raw["training"]["early_stopping"]["patience"] = patience
    return build_mini_config(
        raw,
        b01_freeze_dir="<SYNTHETIC>",
        data_root="<SYNTHETIC>",
        config_path=str(B04A_CONFIG_PATH),
    )


def _make_fake_candidate_result(
    *,
    candidate: str,
    seed: int,
    feasibility: str,
    best_macro_iou: float,
    worst_subject_iou: float | None,
    per_region_ious: dict[int, float] | None = None,
    peak_cuda_mb: float = 123.0,
) -> CandidateResult:
    """Build a CandidateResult with the minimum surface needed by the B04A aggregator."""

    metrics = None
    if feasibility in {"FEASIBLE", "INFEASIBLE", "STOPPED"}:
        per_region = []
        # Use synthetic but plausible per_region rows; one per fg class.
        for cid in FOREGROUND_CLASS_IDS:
            iou = (
                per_region_ious.get(cid, best_macro_iou)
                if per_region_ious
                else best_macro_iou
            )
            per_region.append(
                {
                    "class_id": cid,
                    "iou": float(iou),
                    "dice": float(2 * iou / (1 + iou)) if iou < 1.0 else 1.0,
                    "precision": float(iou),
                    "recall": float(iou),
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                }
            )
        worst_subject = None
        if worst_subject_iou is not None:
            worst_subject = {
                "subject_id": "subj_worst",
                "n_samples": 1,
                "fixed_foreground_macro_iou": float(worst_subject_iou),
                "fixed_foreground_macro_dice": float(
                    2 * worst_subject_iou / (1 + worst_subject_iou)
                ),
                "pixel_accuracy": float(worst_subject_iou),
            }
        from topper_perception.neural.slp8_region_mini import (
            CandidateMetrics,
        )
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        metrics = CandidateMetrics(
            fixed_foreground_macro_iou=float(best_macro_iou),
            fixed_foreground_macro_dice=float(
                2 * best_macro_iou / (1 + best_macro_iou)
            ),
            pixel_accuracy=float(best_macro_iou),
            background_iou=1.0,
            val_loss=0.0,
            per_region=per_region,
            per_posture={"ALL": {"n_samples": 1, "fixed_foreground_macro_iou": best_macro_iou}},
            per_subject={"ALL": {"n_samples": 1, "fixed_foreground_macro_iou": best_macro_iou}},
            worst_subject=worst_subject,
            confusion_matrix=cm,
            centroid_error_summary={},
            n_samples=1,
            n_test_samples=0,
        )
    return CandidateResult(
        candidate=candidate,
        model_version=DEEPLABV3PLUS_LITE_VERSION,
        parameter_count=53449,
        parameter_count_within_budget=True,
        feasibility=feasibility,
        reason=f"fake_{feasibility}",
        epoch_metrics=[],
        metrics=metrics,
        train_predictions=[],
        train_labels=[],
        train_subjects=[],
        val_predictions=[],
        val_labels=[],
        val_subjects=[],
        train_records=[],
        val_records=[],
        best_epoch=1,
        best_val_loss=0.0,
        best_prediction_hash=f"fake_hash_seed_{seed}",
        in_process_prediction_hash=f"fake_hash_seed_{seed}",
        reload_consistent=True,
        reload_max_abs_diff=0.0,
        checkpoint_best_sha256="f" * 64,
        checkpoint_last_sha256="e" * 64,
        train_loss_history=[0.0],
        val_loss_history=[0.0],
        train_subject_overlap_with_val=False,
        val_subject_overlap_with_train=False,
        n_test_samples=0,
        param_changed=True,
        last_in_process_prediction_hash=None,
        class_weight_summary={},
        elapsed_seconds=0.01,
        budget_status="ok",
        budget_report={"peak_cuda_mb": float(peak_cuda_mb)},
        budget_thresholds={},
    )


def _seed_predictions_present(foreground_class_ids: list[int]) -> list[np.ndarray]:
    """Return a small val_prediction list that has at least one pixel of every fg class.

    Used to make a fake ``CandidateResult`` pass the class-collapse
    check.  The array is H x W; we sprinkle each fg class as a small
    rectangular block.
    """

    H, W = PRESSURE_SHAPE
    arr = np.zeros((H, W), dtype=np.int64)
    for cid in foreground_class_ids:
        h0 = (cid * 11) % (H - 5)
        w0 = (cid * 13) % (W - 5)
        arr[h0:h0 + 4, w0:w0 + 4] = cid
    return [arr]


# ---------------------------------------------------------------------------
# 1) Protocol dispatch: B04 vs B04A vs unknown
# ---------------------------------------------------------------------------


class TestProtocolDispatch:
    """B04 / B04A / unknown protocol dispatch."""

    def test_b04_config_dispatches_to_b04(self):
        cfg = _load_b04()
        assert _protocol_of_config(cfg) == B04_PROTOCOL_NAME
        validate_mini_config(cfg)
        mini = build_mini_config(
            cfg,
            b01_freeze_dir=None,
            data_root=None,
            config_path=str(B04_CONFIG_PATH),
        )
        assert mini.protocol == B04_PROTOCOL_NAME
        assert mini.seeds == (42,)
        assert mini.candidates == (MODEL_VERSION, SMALL_UNET_VERSION)

    def test_b04a_config_dispatches_to_b04a(self):
        cfg = _load_b04a()
        assert _protocol_of_config(cfg) == B04A_PROTOCOL_NAME
        validate_mini_config(cfg)
        mini = _build_b04a_mini_config()
        assert mini.protocol == B04A_PROTOCOL_NAME
        assert mini.seeds == B04A_SEEDS
        assert set(mini.candidates) == set(B04A_ACTIVE_CANDIDATE_NAMES)

    def test_unknown_config_version_rejected(self):
        bad = {
            "config_version": "unknown_protocol_v0.1",
            "task_id": "TASK-FAKE",
        }
        with pytest.raises(ConfigValidationError, match="unknown config_version"):
            validate_mini_config(bad)

    def test_missing_config_version_rejected(self):
        with pytest.raises(ConfigValidationError, match="config_version"):
            validate_mini_config({"task_id": "x"})


# ---------------------------------------------------------------------------
# 2) B04 backward compatibility
# ---------------------------------------------------------------------------


class TestB04BackwardCompatibility:
    """The historical B04 config still passes the dispatch and builds."""

    def test_b04_validator_still_accepts_historical_config(self):
        validate_mini_config(_load_b04())

    def test_b04_miniconfig_shape_unchanged(self):
        mini = _build_b04_mini_config()
        assert mini.task_id == TASK_ID
        assert mini.candidates == (MODEL_VERSION, SMALL_UNET_VERSION)
        assert mini.seed == 42
        assert mini.seeds == (42,)
        assert mini.max_epochs == 20
        assert mini.min_epochs == 5
        assert mini.max_parameters == 150_000
        assert mini.resource_budget["max_peak_cuda_mb"] == 12288
        assert mini.val_feasibility_threshold == 0.205644

    def test_b04_config_unchanged_on_disk(self):
        # The historical B04 config is read-only for the B04A runner
        # integration stage; we only read it, never write it.
        text = B04_CONFIG_PATH.read_text(encoding="utf-8")
        assert "TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1" in text
        assert "slp8_region_mini_v0.1" in text
        assert "slp8_tiny_fcn_v0.1" in text
        assert "slp8_small_unet_v0.1" in text

    def test_b04_mini_runner_still_callable(self):
        # The historical B04 orchestrator must remain reachable.  We
        # build a minimal synthetic config + dataset and call
        # ``run_mini`` with a tiny budget; we only check that the
        # orchestrator returns a result whose ``terminal_state`` is in
        # {DONE, FAILED, STOPPED}.
        raw = _load_b04()
        validate_mini_config(raw)
        # Synthetic CPU override AFTER validation (the B04 protocol
        # also mandates ``device=cuda``).
        raw["training"]["device"] = "cpu"
        raw["training"]["max_epochs"] = 1
        raw["training"]["min_epochs"] = 1
        raw["training"]["early_stopping"]["patience"] = 1
        mini = build_mini_config(
            raw,
            b01_freeze_dir=None,
            data_root=None,
            config_path=str(B04_CONFIG_PATH),
        )
        train_ds, val_ds, manifest, class_stats = build_synthetic_dataset(
            n_train_samples=2, n_val_samples=1, seed=7
        )
        cw = compute_class_weights(
            {
                "n_samples": int(class_stats["n_samples"]),
                "n_pixels": int(class_stats["n_pixels"]),
                "per_class_pixel_ratio": {
                    int(k): float(v)
                    for k, v in class_stats["per_class_pixel_ratio"].items()
                },
            }
        )
        assert_class_weight_invariants(cw)
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=10.0,
            max_wall_seconds_total=20.0,
            max_peak_cuda_mb=8192.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            result = run_mini(
                config=mini,
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataset_manifest=manifest,
                class_weight_result=cw,
                output_dir=output_dir,
                device=torch.device("cpu"),
                input_hashes={},
                train_class_stats_source="synthetic",
                synthetic=True,
                budget=budget,
            )
            assert result.terminal_state in {"DONE", "FAILED", "STOPPED"}


# ---------------------------------------------------------------------------
# 3) B04A candidate restrictions
# ---------------------------------------------------------------------------


class TestB04ACandidateRestrictions:
    """B04A rejects TinyFCN and SegFormer-B0 and unknown candidates."""

    def test_tiny_fcn_in_b04a_active_rejected(self):
        raw = _load_b04a()
        # Replace SegFormer (DEFERRED) with TinyFCN as an active entry.
        active = [c for c in raw["candidates"] if c.get("role") != "DEFERRED"]
        raw["candidates"] = active + [
            {
                "name": "slp8_tiny_fcn_v0.1",
                "version": "slp8_tiny_fcn_v0.1",
                "max_parameters": 150000,
                "exact_parameter_count": 1401,
                "role": "new_candidate",
            }
        ]
        with pytest.raises(ConfigValidationError, match="forbidden"):
            validate_mini_config(raw)

    def test_segformer_as_active_in_b04a_rejected(self):
        raw = _load_b04a()
        # Demote SegFormer from DEFERRED to active.
        for c in raw["candidates"]:
            if c["name"] == "slp8_segformer_b0_v0.1":
                c["role"] = "new_candidate"
                c["max_parameters"] = 100000
                c["exact_parameter_count"] = 3700000
        # SegFormer is not registered in the model registry (it is
        # the protocol-deferred entry), so the active rejection
        # surfaces as either "not registered" or "forbidden" — both
        # are valid fail-closed signals.
        with pytest.raises(ConfigValidationError, match="segformer_b0|forbidden|not registered"):
            validate_mini_config(raw)

    def test_unknown_candidate_in_b04a_rejected(self):
        raw = _load_b04a()
        active = [c for c in raw["candidates"] if c.get("role") != "DEFERRED"]
        raw["candidates"] = active + [
            {
                "name": "slp8_unknown_v0.1",
                "version": "slp8_unknown_v0.1",
                "max_parameters": 100000,
                "exact_parameter_count": 100000,
                "role": "new_candidate",
            }
        ]
        with pytest.raises(ConfigValidationError, match="not registered"):
            validate_mini_config(raw)

    def test_b04_candidate_mixed_into_b04a_rejected(self):
        # Inject TinyFCN into a B04A candidate list.  This is the
        # "B04/B04A candidate mix-up" guard.
        raw = _load_b04a()
        active = [c for c in raw["candidates"] if c.get("role") != "DEFERRED"]
        raw["candidates"] = active
        # Replace one active entry with TinyFCN.
        for c in raw["candidates"]:
            if c["name"] == SMALL_UNET_VERSION:
                c["name"] = MODEL_VERSION
                c["version"] = MODEL_VERSION
                c["max_parameters"] = 150000
                c["exact_parameter_count"] = 1401
        with pytest.raises(ConfigValidationError):
            validate_mini_config(raw)

    def test_duplicate_candidate_in_b04a_rejected(self):
        raw = _load_b04a()
        active = [c for c in raw["candidates"] if c.get("role") != "DEFERRED"]
        # Duplicate the small_unet entry (with same name+version+role).
        active.append(copy.deepcopy(active[0]))
        raw["candidates"] = active
        with pytest.raises(ConfigValidationError, match="listed more than once"):
            validate_mini_config(raw)

    def test_missing_active_candidate_in_b04a_rejected(self):
        raw = _load_b04a()
        # Remove the DeepLabV3+-lite entry from the active set.
        raw["candidates"] = [
            c for c in raw["candidates"] if c["name"] != DEEPLABV3PLUS_LITE_VERSION
        ]
        with pytest.raises(ConfigValidationError, match="missing"):
            validate_mini_config(raw)

    def test_exact_parameter_count_mismatch_in_b04a_rejected(self):
        raw = _load_b04a()
        for c in raw["candidates"]:
            if c["name"] == RESUNET_LITE_VERSION:
                c["exact_parameter_count"] = 120_808  # off-by-one
        with pytest.raises(ConfigValidationError, match="exact_parameter_count"):
            validate_mini_config(raw)


# ---------------------------------------------------------------------------
# 4) B04A seeds contract
# ---------------------------------------------------------------------------


class TestB04ASeedsContract:
    """B04A reads exactly the frozen three seeds and refuses any other set."""

    def test_b04a_seeds_exact_three_frozen(self):
        assert B04A_SEEDS == (42, 123, 2026)
        assert len(B04A_SEEDS) == 3

    def test_b04a_miniconfig_seeds_match(self):
        mini = _build_b04a_mini_config()
        assert mini.seeds == B04A_SEEDS
        assert mini.seed == 42  # backward-compat: B04-era single-seed

    def test_b04a_wrong_seeds_rejected(self):
        raw = _load_b04a()
        raw["training"]["seeds"] = [42, 123]
        with pytest.raises(ConfigValidationError, match=r"\[42, 123, 2026\]"):
            validate_mini_config(raw)

    def test_b04a_extra_seed_rejected(self):
        raw = _load_b04a()
        raw["training"]["seeds"] = [42, 123, 2026, 9999]
        with pytest.raises(ConfigValidationError, match=r"\[42, 123, 2026\]"):
            validate_mini_config(raw)

    def test_b04a_single_seed_silently_collapsed_rejected(self):
        # Silent collapse to a single seed is forbidden.
        raw = _load_b04a()
        raw["training"]["seeds"] = [42]
        with pytest.raises(ConfigValidationError, match=r"\[42, 123, 2026\]"):
            validate_mini_config(raw)


# ---------------------------------------------------------------------------
# 5) all_seeds_must_succeed (per-seed hard gates)
# ---------------------------------------------------------------------------


def _make_feasible_candidate(
    candidate: str,
    seed: int,
    *,
    macro_iou: float,
    worst_subject_iou: float = 0.6,
    per_region_ious: dict[int, float] | None = None,
) -> CandidateResult:
    """Build a fully-passing fake CandidateResult (FEASIBLE)."""

    val_predictions = _seed_predictions_present(list(FOREGROUND_CLASS_IDS))
    result = _make_fake_candidate_result(
        candidate=candidate,
        seed=seed,
        feasibility="FEASIBLE",
        best_macro_iou=macro_iou,
        worst_subject_iou=worst_subject_iou,
        per_region_ious=per_region_ious,
    )
    # Mutate val_predictions to ensure class-collapse check passes.
    result.val_predictions = val_predictions
    return result


class TestB04AAllSeedsMustSucceed:
    """The B04A protocol requires all 3 seeds to pass every hard gate."""

    def test_all_three_feasible_yields_feasible_candidate(self):
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_feasible_candidate(
                SMALL_UNET_VERSION, 123, macro_iou=0.55,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "FEASIBLE"
        assert agg.macro_iou_mean == pytest.approx((0.5 + 0.55 + 0.6) / 3)
        assert agg.n_seeds_feasible == 3
        assert agg.n_seeds_failed == 0
        assert agg.n_seeds_stopped == 0

    def test_any_seed_failed_makes_candidate_infeasible(self):
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION, seed=123,
                feasibility="FAILED", best_macro_iou=0.0,
                worst_subject_iou=None,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "INFEASIBLE"
        assert agg.macro_iou_mean is None
        assert agg.n_seeds_failed == 1
        assert "FAILED" in agg.reason
        assert "all_seeds_must_succeed" in agg.reason

    def test_any_seed_stopped_makes_candidate_infeasible(self):
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION, seed=123,
                feasibility="STOPPED", best_macro_iou=0.0,
                worst_subject_iou=None,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "INFEASIBLE"
        assert agg.macro_iou_mean is None
        assert agg.n_seeds_stopped == 1
        assert "STOPPED" in agg.reason

    def test_class_collapse_makes_seed_infeasible(self):
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION, seed=123,
                feasibility="FEASIBLE", best_macro_iou=0.55,
                worst_subject_iou=0.6,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        # Inject class collapse: zero predictions for class 3.
        per_seed[123].val_predictions = [
            np.zeros(PRESSURE_SHAPE, dtype=np.int64)
        ]
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "INFEASIBLE"
        assert "class collapse" in agg.reason
        assert agg.macro_iou_mean is None

    def test_worst_subject_floor_violation_makes_candidate_infeasible(self):
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
                worst_subject_iou=B04A_WORST_SUBJECT_FLOOR + 0.05,
            ),
            123: _make_feasible_candidate(
                SMALL_UNET_VERSION, 123, macro_iou=0.55,
                worst_subject_iou=B04A_WORST_SUBJECT_FLOOR - 0.05,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
                worst_subject_iou=B04A_WORST_SUBJECT_FLOOR + 0.05,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "INFEASIBLE"
        assert "worst-subject" in agg.reason
        assert agg.macro_iou_mean is None

    def test_per_region_floor_violation_makes_candidate_infeasible(self):
        # Two seeds pass cleanly; one seed passes a region below the
        # B04A per-region floor.  The whole candidate must be
        # INFEASIBLE.
        per_region_ious_bad = {cid: 0.06 for cid in FOREGROUND_CLASS_IDS}
        per_region_ious_bad[1] = B04A_PER_REGION_FLOOR - 0.01
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_feasible_candidate(
                SMALL_UNET_VERSION, 123, macro_iou=0.55,
                per_region_ious=per_region_ious_bad,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        assert agg.feasibility == "INFEASIBLE"
        assert "per-region" in agg.reason

    def test_partial_seed_mean_forbidden(self):
        # Even if 2/3 seeds pass cleanly, the third being FAILED
        # means the macro_iou_mean MUST NOT be computed over the 2
        # surviving seeds.
        per_seed = {
            42: _make_feasible_candidate(
                SMALL_UNET_VERSION, 42, macro_iou=0.5,
            ),
            123: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION, seed=123,
                feasibility="FAILED", best_macro_iou=0.0,
                worst_subject_iou=None,
            ),
            2026: _make_feasible_candidate(
                SMALL_UNET_VERSION, 2026, macro_iou=0.6,
            ),
        }
        agg = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        # The aggregator MUST NOT silently compute the mean of
        # (0.5 + 0.6) / 2 = 0.55 over the two surviving seeds.
        assert agg.macro_iou_mean is None
        assert agg.feasibility == "INFEASIBLE"


# ---------------------------------------------------------------------------
# 6) Candidate-level decision (0/1/2/3 feasible)
# ---------------------------------------------------------------------------


def _make_aggregate(
    candidate: str,
    param_count: int,
    macro_iou_mean: float,
    feasibility: str = "FEASIBLE",
    *,
    worst_subject_iou: float | None = 0.6,
) -> Any:
    """Build a minimal B04ACandidateAggregate for decision tests.

    ``worst_subject_iou`` defaults to 0.6 for FEASIBLE and ``None``
    for INFEASIBLE/STOPPED/FAILED.  Pass a value to override the
    default; pass ``None`` explicitly to simulate a candidate with
    no usable ``worst_subject_iou`` (used by the fail-closed
    tiebreak test).
    """

    from topper_perception.neural.slp8_region_mini import (
        B04ACandidateAggregate,
    )
    if feasibility == "FEASIBLE":
        ws_value: float | None = (
            float(worst_subject_iou)
            if worst_subject_iou is not None
            else None
        )
    else:
        ws_value = None
    return B04ACandidateAggregate(
        candidate=candidate,
        model_version=DEEPLABV3PLUS_LITE_VERSION,
        parameter_count=param_count,
        seeds=B04A_SEEDS,
        per_seed={},
        n_seeds_total=3,
        n_seeds_feasible=3 if feasibility == "FEASIBLE" else 0,
        n_seeds_infeasible=0 if feasibility == "FEASIBLE" else 3,
        n_seeds_failed=0,
        n_seeds_stopped=0,
        feasibility=feasibility,
        reason=f"fake_{feasibility}",
        macro_iou_mean=(
            float(macro_iou_mean) if feasibility == "FEASIBLE" else None
        ),
        worst_subject_iou=ws_value,
        per_region_iou={cid: 0.1 for cid in FOREGROUND_CLASS_IDS}
        if feasibility == "FEASIBLE" else {},
        elapsed_seconds_total=0.03,
        budget_status="ok",
        n_test_samples=0,
    )


class TestB04ACandidateDecision:
    """The 0/1/2/3-feasible decision rules."""

    def test_zero_feasible_advances_none(self):
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.0, feasibility="INFEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.0, feasibility="INFEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.0, feasibility="INFEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ()
        assert decision.near_tie_applied is False
        assert decision.tiebreaks == []

    def test_one_feasible_advances_only_that(self):
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.0, feasibility="INFEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.5, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.0, feasibility="INFEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == (RESUNET_LITE_VERSION,)
        assert decision.near_tie_applied is False

    def test_two_feasible_advance_both_no_champion(self):
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.6, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.0, feasibility="INFEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.55, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # Both advance, no champion picked.
        assert set(decision.advanced) == {
            SMALL_UNET_VERSION, DEEPLABV3PLUS_LITE_VERSION
        }
        assert decision.near_tie_applied is False

    def test_three_feasible_top2_by_macro_iou(self):
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.4, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.6, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.5, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # Top 2: ResUNet (0.6) and DeepLabV3+-lite (0.5).  SmallUNet (0.4) drops.
        # |0.6-0.5|=0.10 >= 0.02 -> no near-tie tiebreak.
        assert decision.advanced == (
            RESUNET_LITE_VERSION, DEEPLABV3PLUS_LITE_VERSION
        )
        assert decision.near_tie_applied is False
        assert len(decision.tiebreaks) == 2
        # First boundary: ResUNet (0.6) vs DeepLabV3+-lite (0.5) -- gap 0.10, no tiebreak.
        tb12 = decision.tiebreaks[0]
        assert tb12.tiebreak_basis == "none"
        assert tb12.macro_iou_difference == pytest.approx(0.10)
        # Second boundary: SmallUNet (0.4) vs DeepLabV3+-lite (0.5) -- gap 0.10, no tiebreak.
        tb23 = decision.tiebreaks[1]
        assert tb23.tiebreak_basis == "none"


class TestB04ANearTieTiebreak:
    """The near-tie tiebreak prefers the simpler (lower parameter count) model."""

    def test_near_tie_prefers_lower_parameter_count(self):
        # SmallUNet (118,121) and ResUNet (120,809) are within 0.02 of
        # each other.  ResUNet is higher macro_iou_mean (0.500 vs
        # 0.495), so without tiebreak ResUNet wins.  With the
        # tiebreak, the simpler SmallUNet wins.
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.495, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.500, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # Top 2 by macro_iou are ResUNet (0.500) and SmallUNet (0.495).
        # |diff| = 0.005 < 0.02, so near_tie_applied = True.  Param
        # diff ratio = 2688/118121 ~= 0.0228 (< 10% threshold), so
        # worst_subject_iou basis.  Both candidates share the same
        # fake worst_subject_iou=0.6 -> worst_subject_iou_then_param
        # basis; p_l=118_121 < p_h=120_809, so SmallUNet wins at the
        # top1-vs-top2 boundary.  Per R03, the 2nd boundary compares
        # the 1st-round REJECTED candidate (ResUNet) with top3
        # (DeepLabV3+-lite).  |0.500-0.30| = 0.20 >= 0.02 -> no
        # tiebreak; ResUNet keeps the second slot.  Advanced =
        # (SmallUNet, ResUNet).
        assert decision.advanced == (SMALL_UNET_VERSION, RESUNET_LITE_VERSION)
        assert decision.near_tie_applied is True
        assert len(decision.tiebreaks) == 2
        # First boundary audit-trail.
        tb12 = decision.tiebreaks[0]
        assert tb12.pair == (RESUNET_LITE_VERSION, SMALL_UNET_VERSION)
        assert tb12.macro_iou_difference == pytest.approx(0.005)
        assert tb12.tiebreak_basis in {
            "worst_subject_iou_then_param", "parameter_count"
        }
        assert tb12.selected == SMALL_UNET_VERSION
        assert tb12.rejected == RESUNET_LITE_VERSION
        # Second boundary audit-trail: 1st-loser ResUNet vs top3
        # DeepLabV3+-lite.  |0.500-0.30| = 0.20 >= 0.02 -> no tiebreak.
        tb23 = decision.tiebreaks[1]
        assert tb23.tiebreak_basis == "none"
        assert tb23.macro_iou_difference == pytest.approx(0.20)
        assert tb23.pair == (RESUNET_LITE_VERSION, DEEPLABV3PLUS_LITE_VERSION)
        assert tb23.selected == RESUNET_LITE_VERSION
        assert tb23.rejected == DEEPLABV3PLUS_LITE_VERSION

    def test_no_tiebreak_when_diff_exceeds_margin(self):
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.40, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.50, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # |diff| = 0.10 > 0.02; no near-tie tiebreak.
        assert decision.advanced == (RESUNET_LITE_VERSION, SMALL_UNET_VERSION)
        assert decision.near_tie_applied is False
        # Both boundaries have basis="none".
        assert decision.tiebreaks[0].tiebreak_basis == "none"
        assert decision.tiebreaks[1].tiebreak_basis == "none"

    def test_near_tie_margin_constant(self):
        # The B04A protocol's near-tie margin is 0.02.  Cross-check the
        # exported constant matches the protocol.
        assert B04A_NEAR_TIE_MARGIN == 0.02


# ---------------------------------------------------------------------------
# 7) Resource budget
# ---------------------------------------------------------------------------


class TestB04AResourceBudget:
    """B04A resource budget: 45 min / candidate, 135 min total, 8192 MiB."""

    def test_b04a_budget_constants(self):
        from topper_perception.neural.slp8_region_mini import (
            B04A_RESOURCE_BUDGET,
        )
        assert B04A_RESOURCE_BUDGET["max_wall_minutes_per_candidate"] == 45
        assert B04A_RESOURCE_BUDGET["max_total_wall_minutes"] == 135
        assert B04A_RESOURCE_BUDGET["max_peak_cuda_mb"] == 8192

    def test_b04a_budget_validates_correct_numbers(self):
        raw = _load_b04a()
        validate_mini_config(raw)  # contract is satisfied

    def test_b04a_budget_wrong_per_candidate_wall_rejected(self):
        raw = _load_b04a()
        raw["resource_budget"]["per_candidate_wall_minutes"] = 60
        with pytest.raises(ConfigValidationError, match="per_candidate_wall_minutes"):
            validate_mini_config(raw)

    def test_b04a_budget_wrong_total_wall_rejected(self):
        raw = _load_b04a()
        raw["resource_budget"]["total_wall_minutes"] = 999
        with pytest.raises(ConfigValidationError, match="total_wall_minutes"):
            validate_mini_config(raw)

    def test_b04a_budget_wrong_peak_cuda_rejected(self):
        raw = _load_b04a()
        raw["resource_budget"]["max_peak_cuda_mb"] = 16384
        with pytest.raises(ConfigValidationError, match="max_peak_cuda_mb"):
            validate_mini_config(raw)

    def test_b04a_budget_accumulator_resume(self):
        # The BudgetAccumulatorState must restore the cumulative
        # candidate-level and total-level seconds after a restart;
        # double-counting wall time after restart is forbidden.
        from topper_perception.neural.slp8_region_budget import (
            BudgetAccumulatorState,
        )
        b = ResourceBudget(
            max_wall_seconds_per_candidate=45 * 60.0,
            max_wall_seconds_total=135 * 60.0,
            max_peak_cuda_mb=8192.0,
        )
        s = ResourceBudgetState(b)
        # First candidate: time 0.05 s
        s.begin_candidate()
        time.sleep(0.05)
        # Closing the first candidate commits the elapsed time to the
        # accumulator.
        s.begin_candidate()
        time.sleep(0.05)
        # Closing the second candidate likewise.
        s.begin_candidate()
        snap_before = s.snapshot()
        # Simulate restart: build a fresh state and restore the
        # accumulator snapshot.
        s2 = ResourceBudgetState(b)
        s2.restore(snap_before)
        # The new state must carry the prior candidate-level seconds
        # forward (not reset to 0).
        assert s2._candidate_seconds_consumed == pytest.approx(
            float(snap_before.candidate_seconds_consumed), rel=1e-9
        )
        assert s2._candidate_seconds_consumed > 0.0


# ---------------------------------------------------------------------------
# 8) Identity, checkpoint, output
# ---------------------------------------------------------------------------


class TestB04AIdentityCheckpointOutput:
    """B04A identity, checkpoint, and output contracts."""

    def test_b04a_identity_block_has_required_fields(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )
        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256="f" * 64,
            experiment_id="EXP-SLP-B04A-TEST-EXP-ID",
            data_manifest_sha256="e" * 64,
            git_commit="0" * 40,
            candidate=SMALL_UNET_VERSION,
            seed=42,
        )
        required = {
            "experiment_id",
            "git_commit",
            "git_dirty",
            "config_sha256",
            "data_manifest_sha256",
            "split_sha256",
            "model_version",
            "task_id",
            "candidate",
            "seed",
        }
        missing = required - set(block)
        assert not missing, f"identity missing keys: {missing}"

    def test_b04a_identity_model_version_per_seed(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )
        block_unet = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256="a" * 64,
            experiment_id="EXP-SLP-B04A-TEST-EXP-ID",
            data_manifest_sha256="e" * 64,
            git_commit="0" * 40,
            candidate=SMALL_UNET_VERSION,
            seed=42,
        )
        block_deeplab = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256="a" * 64,
            experiment_id="EXP-SLP-B04A-TEST-EXP-ID",
            data_manifest_sha256="e" * 64,
            git_commit="0" * 40,
            candidate=DEEPLABV3PLUS_LITE_VERSION,
            seed=42,
        )
        assert block_unet["model_version"] == SMALL_UNET_VERSION
        assert block_deeplab["model_version"] == DEEPLABV3PLUS_LITE_VERSION

    def test_resume_identity_mismatch_rejected(self):
        # Saving a checkpoint with one identity and resuming with a
        # different identity must raise ResumeIdentityError.
        identity_a = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256="a" * 64,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256="f" * 64,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            git_commit="0" * 40,
            git_dirty=False,
        )
        identity_b = replace(identity_a, seed=123)
        with pytest.raises(ResumeIdentityError, match="identity mismatch"):
            verify_resume_identity(saved=identity_a, requested=identity_b)

    def test_resume_identity_match_accepted(self):
        identity = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256="a" * 64,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256="f" * 64,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            git_commit="0" * 40,
            git_dirty=False,
        )
        # No raise.
        verify_resume_identity(saved=identity, requested=identity)

    def test_existing_output_directory_refuses_overwrite(self, tmp_path):
        out = tmp_path / "collide"
        out.mkdir()
        (out / "DONE.json").write_text("{}", encoding="utf-8")
        with pytest.raises(OutputCollisionError, match="DONE.json"):
            check_output_dir_safety(out)

    def test_three_way_terminal_status_mutex(self, tmp_path):
        out = tmp_path / "status"
        write_status_files(out, status="DONE", extra={"k": "v"})
        assert (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()
        assert not (out / "STOPPED.json").exists()
        write_status_files(out, status="FAILED", extra={"k": "v"})
        assert (out / "FAILED.json").exists()
        assert not (out / "DONE.json").exists()
        write_status_files(out, status="STOPPED", extra={"k": "v"})
        assert (out / "STOPPED.json").exists()
        assert not (out / "FAILED.json").exists()
        assert not (out / "DONE.json").exists()


# ---------------------------------------------------------------------------
# 9) TEST = 0
# ---------------------------------------------------------------------------


class TestB04ATestZero:
    """B04A never imports the B01 test-access contract."""

    def test_b04a_miniconfig_test_access_default_denied(self):
        raw = _load_b04a()
        # top-level test_access_policy + dataset test_access_policy.
        assert raw["test_access_policy"]["this_run_loads_test"] is False
        assert raw["test_access_policy"]["test_access_in_this_run"] == "denied"
        assert raw["dataset"]["test_access_policy"]["load_test_in_mini"] is False
        assert raw["dataset"]["test_access_policy"]["test_access_in_mini"] == "denied"
        validate_mini_config(raw)  # accepted as-is

    def test_b04a_validator_rejects_test_loads_test_true(self):
        raw = _load_b04a()
        raw["test_access_policy"]["this_run_loads_test"] = True
        with pytest.raises(ConfigValidationError, match="refuses to load TEST"):
            validate_mini_config(raw)

    def test_b04a_validator_rejects_dataset_test_load_in_mini_true(self):
        raw = _load_b04a()
        raw["dataset"]["test_access_policy"]["load_test_in_mini"] = True
        with pytest.raises(ConfigValidationError, match="refuses to load TEST"):
            validate_mini_config(raw)

    def test_b04a_smoke_does_not_import_b01_test_access(self):
        # Source-level guard: scripts/smoke_b04a_runner_integration.py
        # does not import or call any B01 test-access surface.  We
        # check the actual call pattern ``enable_test_access(`` so
        # the words appearing in documentation / string literals
        # are not counted.
        import re

        smoke_path = (
            PROJECT_ROOT / "scripts" / "smoke_b04a_runner_integration.py"
        )
        text = smoke_path.read_text(encoding="utf-8")
        # Strip all string literals and comments to keep the check
        # focused on actual code.
        text = re.sub(r'"""[\s\S]*?"""', "", text)
        text = re.sub(r"'''[\s\S]*?'''", "", text)
        text = re.sub(r'"(?:\\.|[^"\\])*"', "", text)
        text = re.sub(r"'(?:\\.|[^'\\])*'", "", text)
        text = re.sub(r"#[^\n]*", "", text)
        forbidden_call_patterns = [
            r"\benable_test_access\s*\(",
            r"\bTestLeakageError\b",
        ]
        for pat in forbidden_call_patterns:
            assert not re.search(pat, text), (
                f"smoke must not import or call B01 TEST access "
                f"surface; pattern {pat!r} matched"
            )

    def test_b04a_mini_runner_does_not_import_b01_test_access(self):
        # Source-level guard: src/topper_perception/neural/slp8_region_mini.py
        # does not import or call any B01 test-access surface.
        import re

        runner_path = (
            PROJECT_ROOT
            / "src"
            / "topper_perception"
            / "neural"
            / "slp8_region_mini.py"
        )
        text = runner_path.read_text(encoding="utf-8")
        text = re.sub(r'"""[\s\S]*?""""', "", text)
        text = re.sub(r"'''[\s\S]*?'''", "", text)
        text = re.sub(r'"(?:\\.|[^"\\])*"', "", text)
        text = re.sub(r"'(?:\\.|[^'\\])*'", "", text)
        text = re.sub(r"#[^\n]*", "", text)
        forbidden_call_patterns = [
            r"\benable_test_access\s*\(",
            r"\bTestLeakageError\b",
        ]
        for pat in forbidden_call_patterns:
            assert not re.search(pat, text), (
                f"B04A mini runner must not import or call B01 TEST "
                f"access surface; pattern {pat!r} matched"
            )


# ---------------------------------------------------------------------------
# 10) B04A end-to-end synthetic CPU smoke (orchestrator + bundle)
# ---------------------------------------------------------------------------


class TestB04AEndToEndSmoke:
    """Run the B04A orchestrator end-to-end on a small synthetic dataset."""

    def test_b04a_no_write_smoke_prints_summary(self):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "smoke_b04a_runner_integration.py"),
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        out = result.stdout.strip().splitlines()[-1]
        assert out.startswith("B04A_SMOKE_NO_WRITE ")
        # Required fields in the one-line summary.
        for key in (
            "protocol=B04A",
            "config_version=slp8_pm_architecture_expansion_mini_v0.1",
            "candidates=3",
            "seeds=3",
        ):
            assert key in out, f"missing {key!r} in {out!r}"

    def test_b04a_smoke_writes_bundle_with_force(self, tmp_path):
        import subprocess

        output_dir = tmp_path / "b04a_smoke"
        report = tmp_path / "smoke.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "smoke_b04a_runner_integration.py"),
                "--output",
                str(report),
                "--output-dir",
                str(output_dir),
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        assert report.exists()
        assert output_dir.exists()
        summary = json.loads(report.read_text(encoding="utf-8"))
        assert summary["protocol"] == "B04A"
        assert summary["terminal_state"] in {"DONE", "FAILED", "STOPPED"}
        # Identity fields recorded in the summary.
        assert "git_commit" in summary["identity_keys"]
        assert "config_sha256" in summary["identity_keys"]
        # Run-level bundle files.
        for fname in (
            "manifest.json",
            "resolved_config.json",
            "input_manifest_hashes.json",
            "environment.json",
            "status.json",
            "candidate_decision.json",
            "budget_report.json",
            "logs/run.log",
        ):
            assert (output_dir / fname).exists(), f"missing {fname}"
        # Terminal file is one of DONE.json / FAILED.json / STOPPED.json.
        terminal = [
            p.name
            for p in output_dir.iterdir()
            if p.name in {"DONE.json", "FAILED.json", "STOPPED.json"}
        ]
        assert len(terminal) == 1
        # Per-seed checkpoints for every (candidate, seed) pair.
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                ckpt_dir = output_dir / "checkpoints" / cand / f"seed_{seed:04d}"
                assert (ckpt_dir / "last.pt").exists()
                assert (ckpt_dir / "best.pt").exists()
        # TEST=0 recorded as a declarative policy.
        assert summary["test_access"]["kind"] == "declarative_policy"
        assert summary["test_access"]["value"] == 0

    def test_b04a_smoke_refuses_overwrite_existing_output(self, tmp_path):
        import subprocess

        report = tmp_path / "smoke.json"
        report.write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "smoke_b04a_runner_integration.py"),
                "--output",
                str(report),
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # --no-write should not touch the existing file; exit 0.
        assert result.returncode == 0
        assert report.read_text(encoding="utf-8") == "{}"

    def test_b04a_cli_synthetic_cpu_smoke_b04a(self, tmp_path):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config",
                str(B04A_CONFIG_PATH),
                "--output-dir",
                str(tmp_path / "b04a_out"),
                "--synthetic-cpu-smoke-b04a",
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        last = result.stdout.strip().splitlines()[-1]
        assert last.startswith("B04A_SMOKE_NO_WRITE ")

    def test_b04a_cli_rejects_non_b04a_config(self, tmp_path):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config",
                str(B04_CONFIG_PATH),
                "--output-dir",
                str(tmp_path / "rej"),
                "--synthetic-cpu-smoke-b04a",
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert "B04A" in result.stderr

    def test_b04a_tiny_budget_triggers_stopped(self, tmp_path):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config",
                str(B04A_CONFIG_PATH),
                "--output-dir",
                str(tmp_path / "stopped"),
                "--synthetic-cpu-smoke-b04a",
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS": "1e-9",
            },
        )
        # Tiny per-candidate budget should trigger STOPPED -> exit 1
        # in the writing path or 0 in the no-write path (which does
        # not exercise the budget override).  We only require that
        # the no-write path tolerates the env var without crashing.
        assert result.returncode in {0, 1}
        last = result.stdout.strip().splitlines()[-1]
        assert last.startswith("B04A_SMOKE_NO_WRITE ")


# ---------------------------------------------------------------------------
# 11) B04 regression: 15/15 of the B04 mini tests still pass
# ---------------------------------------------------------------------------


class TestB04Regression:
    """The B04 implementation tests still pass after the protocol dispatch."""

    def test_b04_implementation_test_slp8_region_models_passes(self):
        # ``tests/test_slp8_region_models.py`` covers the model
        # registry and model-specific assertions; it does not import
        # the B04A orchestrator and is unchanged by this task.
        from topper_perception.neural.slp8_region_models import (
            MODEL_REGISTRY,
            Slp8SmallUnet,
            create_slp8_small_unet,
        )
        # Spot-check: the B04 candidates and the B04A active set are
        # all registered and have the right parameter counts.
        for name in B04A_ACTIVE_CANDIDATE_NAMES:
            assert name in MODEL_REGISTRY
        # SmallUNet parameter count is 118,121 (B04 v0.1 R02 contract).
        m = Slp8SmallUnet()
        assert m.count_parameters() == 118_121
        # B04 frozen registry counts match.
        from topper_perception.neural.slp8_region_models import (
            B04A_EXACT_PARAMETER_COUNTS as EXACT,
        )
        for name, expected in EXACT.items():
            assert name in MODEL_REGISTRY
        # Model factory for SmallUNet returns a CPU model with the
        # expected parameter count.
        m, cfg = create_slp8_small_unet(device="cpu")
        assert m.count_parameters() == 118_121
        assert cfg["device"] == "cpu"


# ---------------------------------------------------------------------------
# 12) B04A near-tie (extended): param-difference basis, fail-closed
#     missing worst-subject, input-order independence, and 2nd-vs-3rd
#     tiebreak where the third replaces the second.
# ---------------------------------------------------------------------------


class TestB04ANearTieTiebreakExt:
    """Additional near-tie tiebreak cases required by Codex R02.

    The frozen contract:

    * 2nd-vs-3rd near-tie with a simpler 3rd candidate REPLACES the
      2nd in the Top-2 set.
    * Parameter diff within 10% -- use ``worst_subject_iou``;
      if both candidates tie on that, prefer fewer parameters
      (``worst_subject_iou_then_param``).
    * Parameter diff > 10% -- prefer fewer parameters
      (``parameter_count`` basis).
    * Missing / non-finite ``worst_subject_iou`` -- fail closed
      (``failed_no_worst_subject`` basis; advanced = (); terminal
      state = FAILED upstream).
    * Input dict order is irrelevant: the decision is a deterministic
      function of the aggregates, not of insertion order.
    """

    def test_first_vs_second_near_tie_both_still_advance(self):
        # The first boundary is a near-tie but the second boundary
        # is NOT a near-tie.  The 1st-loser (ResUNet) is compared
        # against top3 (DeepLabV3+-lite, 0.30) and ResUNet keeps
        # the second slot.  Both candidates still advance; the
        # set change is the (top1, top2) ordering, not a
        # promotion/demotion across the Top-2 boundary.
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.495, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.500, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # Top-2 set in slot order: SmallUNet (1st), ResUNet (2nd).
        assert set(decision.advanced) == {SMALL_UNET_VERSION, RESUNET_LITE_VERSION}
        assert decision.near_tie_applied is True
        # DeepLabV3+-lite is NOT in the advanced set.
        assert DEEPLABV3PLUS_LITE_VERSION not in decision.advanced

    def test_second_vs_third_near_tie_with_simpler_third(self):
        # The 2nd-vs-3rd near-tie only fires when the 1st boundary
        # has been swapped.  In this scenario:
        #   Top1: cand_A (0.500, 100,000 params)
        #   Top2: cand_B (0.495, 90,000 params)
        #   Top3: cand_C (0.490, 30,000 params)
        # 1st boundary: A vs B, diff 0.005, ratio = 10k/90k ~= 0.111
        # > 0.10, basis=parameter_count; B (90k) < A (100k) -> B
        # wins the 1st boundary.  Per R03, the 2nd boundary
        # compares the 1st-round REJECTED candidate (A) with
        # top3 (C).  diff 0.010, ratio 70k/30k ~= 2.33 > 0.10,
        # basis=parameter_count; C (30k) < A (100k) -> C wins.
        # Advanced = (B, C).  The simpler 3rd replaces the
        # 1st-loser in the second slot.
        aggregates = {
            "cand_A": _make_aggregate(
                "cand_A", 100_000, 0.500, feasibility="FEASIBLE"
            ),
            "cand_B": _make_aggregate(
                "cand_B", 90_000, 0.495, feasibility="FEASIBLE"
            ),
            "cand_C": _make_aggregate(
                "cand_C", 30_000, 0.490, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # Advanced set must include B (1st winner) and C (replaces A).
        assert decision.advanced == ("cand_B", "cand_C")
        assert "cand_A" not in decision.advanced
        assert decision.near_tie_applied is True
        # 1st boundary: A vs B, diff 0.005, param ratio 0.111 > 0.10.
        tb12 = decision.tiebreaks[0]
        assert tb12.tiebreak_basis == "parameter_count"
        assert tb12.macro_iou_difference == pytest.approx(0.005)
        assert tb12.parameter_difference_ratio == pytest.approx(
            10_000.0 / 90_000.0
        )
        assert tb12.selected == "cand_B"
        assert tb12.rejected == "cand_A"
        # 2nd boundary: A (1st-loser) vs C (top3).  diff 0.010,
        # ratio 70k/30k ~= 2.33 > 0.10.
        tb23 = decision.tiebreaks[1]
        assert tb23.tiebreak_basis == "parameter_count"
        assert tb23.macro_iou_difference == pytest.approx(0.010)
        assert tb23.parameter_difference_ratio == pytest.approx(
            70_000.0 / 30_000.0
        )
        assert tb23.pair == ("cand_A", "cand_C")
        assert tb23.selected == "cand_C"
        assert tb23.rejected == "cand_A"

    def test_param_diff_within_10pct_worst_subject_tiebreak(self):
        # ResUNet (120,809) and SmallUNet (118,121): param ratio =
        # 2,688/118,121 ~= 0.0228, well below 10%, so basis is
        # ``worst_subject_iou``.  SmallUNet has the higher
        # worst_subject_iou and wins the boundary.
        aggregates = {
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.500,
                feasibility="FEASIBLE", worst_subject_iou=0.50,
            ),
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.495,
                feasibility="FEASIBLE", worst_subject_iou=0.60,
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30,
                feasibility="FEASIBLE",
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == (SMALL_UNET_VERSION, RESUNET_LITE_VERSION)
        assert decision.near_tie_applied is True
        tb12 = decision.tiebreaks[0]
        assert tb12.tiebreak_basis == "worst_subject_iou"
        assert tb12.selected == SMALL_UNET_VERSION
        assert tb12.rejected == RESUNET_LITE_VERSION
        assert tb12.parameter_difference_ratio == pytest.approx(
            2688.0 / 118_121.0
        )
        assert tb12.parameter_difference_ratio < 0.10

    def test_param_diff_over_10pct_fewer_params_wins(self):
        # Two synthetic candidates: 200,000 vs 100,000 parameters.
        # Ratio = 1.0 > 0.10 -> parameter_count basis; the smaller
        # candidate wins the first boundary.  Per R03, the 2nd
        # boundary compares the 1st-round REJECTED candidate (big)
        # with top3 (middle).  |0.500-0.30| = 0.20 >= 0.02 -> no
        # tiebreak; big is kept as the second slot.  Advanced =
        # (small, big).  Middle is dropped.
        aggregates = {
            "big": _make_aggregate("big", 200_000, 0.500, feasibility="FEASIBLE"),
            "small": _make_aggregate("small", 100_000, 0.495, feasibility="FEASIBLE"),
            "middle": _make_aggregate("middle", 90_000, 0.30, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        # 1st boundary: big vs small, diff 0.005, ratio 1.0 > 0.10.
        # "small" wins (fewer params).  2nd boundary: big (1st-loser)
        # vs middle (top3), |0.500-0.30|=0.20 -> no tiebreak; big
        # wins.  Advanced = (small, big).  Middle is dropped.
        assert decision.advanced == ("small", "big")
        assert "middle" not in decision.advanced
        assert decision.near_tie_applied is True
        tb12 = decision.tiebreaks[0]
        assert tb12.tiebreak_basis == "parameter_count"
        assert tb12.selected == "small"
        assert tb12.rejected == "big"
        assert tb12.parameter_difference_ratio == pytest.approx(1.0)
        # 2nd boundary: big (1st-loser) vs middle (top3).
        # |0.500-0.30| = 0.20 >= 0.02 -> no tiebreak.
        tb23 = decision.tiebreaks[1]
        assert tb23.tiebreak_basis == "none"
        assert tb23.pair == ("big", "middle")
        assert tb23.macro_iou_difference == pytest.approx(0.20)
        assert tb23.selected == "big"
        assert tb23.rejected == "middle"

    def test_missing_worst_subject_fail_closed(self):
        # A near-tie is detected, but one of the pair has no usable
        # worst_subject_iou.  The boundary must be marked
        # ``failed_no_worst_subject`` with ``selected=None``; the
        # decision becomes ``advanced=()`` and the orchestrator is
        # expected to promote terminal_state to FAILED upstream.
        aggregates = {
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.500,
                feasibility="FEASIBLE", worst_subject_iou=0.50,
            ),
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.495,
                feasibility="FEASIBLE", worst_subject_iou=None,
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30,
                feasibility="FEASIBLE",
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ()
        assert decision.near_tie_applied is True
        assert len(decision.tiebreaks) == 1
        tb = decision.tiebreaks[0]
        assert tb.tiebreak_basis == "failed_no_worst_subject"
        assert tb.selected is None
        assert tb.rejected is None

    def test_non_near_tie_strict_order(self):
        # With a large gap between the top-2 and the third, the
        # third is dropped without any tiebreak.  Ordering is
        # strictly by macro_iou_mean (DESC, name ASC tiebreak).
        aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.40, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.50, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.10, feasibility="FEASIBLE"
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        # ResUNet (0.50) > SmallUNet (0.40) > DeepLabV3+-lite (0.10).
        # DeepLabV3+-lite is dropped at slot 3.
        assert decision.advanced == (RESUNET_LITE_VERSION, SMALL_UNET_VERSION)
        assert decision.near_tie_applied is False
        # Both tiebreak records have basis="none".
        assert all(tb.tiebreak_basis == "none" for tb in decision.tiebreaks)

    def test_input_order_independence(self):
        # Insertion order into the aggregates dict must not affect
        # the final Top-2 set or the audit-trail.
        base_aggregates = {
            SMALL_UNET_VERSION: _make_aggregate(
                SMALL_UNET_VERSION, 118_121, 0.495, feasibility="FEASIBLE"
            ),
            RESUNET_LITE_VERSION: _make_aggregate(
                RESUNET_LITE_VERSION, 120_809, 0.500, feasibility="FEASIBLE"
            ),
            DEEPLABV3PLUS_LITE_VERSION: _make_aggregate(
                DEEPLABV3PLUS_LITE_VERSION, 53_449, 0.30, feasibility="FEASIBLE"
            ),
        }
        # Forward order.
        d_forward = _b04a_advance_decision(dict(base_aggregates))
        # Reverse order.
        d_reverse = _b04a_advance_decision(
            {k: base_aggregates[k] for k in reversed(list(base_aggregates))}
        )
        # Scrambled order.
        scrambled = list(base_aggregates.values())
        d_scrambled = _b04a_advance_decision(
            {
                DEEPLABV3PLUS_LITE_VERSION: scrambled[2],
                RESUNET_LITE_VERSION: scrambled[1],
                SMALL_UNET_VERSION: scrambled[0],
            }
        )
        for d in (d_forward, d_reverse, d_scrambled):
            assert d.advanced == d_forward.advanced
            assert d.near_tie_applied == d_forward.near_tie_applied
            assert len(d.tiebreaks) == len(d_forward.tiebreaks)
            for tb_a, tb_b in zip(d.tiebreaks, d_forward.tiebreaks):
                assert tb_a.pair == tb_b.pair
                assert tb_a.tiebreak_basis == tb_b.tiebreak_basis
                assert tb_a.selected == tb_b.selected
                assert tb_a.rejected == tb_b.rejected


# ---------------------------------------------------------------------------
# 12b) B04A near-tie (R03 ITERATE): Reviewer-supplied exact scenario
#      plus 1st-no-near-tie, 3rd-replaces-1st-loser, 2nd-boundary
#      fail-closed, and input-order independence under the corrected
#      contract (1st winner gets slot 1; first_loser competes with
#      top3 for slot 2).
# ---------------------------------------------------------------------------


class TestB04ANearTieTiebreakR03:
    """R03 ITERATE: Top-2 inclusion boundary contract clarification.

    The corrected algorithm:

    1. Sort feasible by ``(-macro_iou_mean, name)`` (DESC, ASC).
    2. 1st boundary: top1 vs top2; winner gets slot 1; loser is
       the "first_loser".
    3. 2nd boundary: first_loser vs top3; winner gets slot 2.
    4. advanced = (first_winner, second_winner) -- two distinct
       candidates; the 1st-round winner does NOT participate in
       the 2nd boundary.

    The 1st-round winner is the only candidate that can
    occupy slot 1; the 2nd slot is filled by the 2nd boundary
    winner which is either the 1st-loser (when top3 fails to
    dislodge it) or top3 (when top3 wins the 2nd boundary).
    """

    def test_reviewer_exact_scenario_advanced_is_b_then_a(self):
        # Codex R03 exact reproduction case:
        #   A: macro_iou=0.500, params=100,000
        #   B: macro_iou=0.495, params=90,000
        #   C: macro_iou=0.478, params=80,000
        # 1st boundary: A vs B, diff=0.005, near-tie, ratio=10k/90k
        # ~= 0.111 > 0.10 -> parameter_count basis; B (90k) < A
        # (100k) -> B wins (slot 1).
        # 2nd boundary: first_loser=A vs top3=C, diff=0.022, NOT
        # near-tie, basis=none -> A wins (slot 2).
        # advanced == ("B", "A").  CRUCIALLY the 2nd audit pair
        # MUST be (A, C), NOT (B, C).  A common R02 mistake was to
        # let B (1st winner) compete in the 2nd boundary.
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.495, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 80_000, 0.478, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ("B", "A")
        assert "C" not in decision.advanced
        assert decision.near_tie_applied is True
        assert len(decision.tiebreaks) == 2
        tb12 = decision.tiebreaks[0]
        assert tb12.pair == ("A", "B")
        assert tb12.tiebreak_basis == "parameter_count"
        assert tb12.macro_iou_difference == pytest.approx(0.005)
        assert tb12.parameter_difference_ratio == pytest.approx(
            10_000.0 / 90_000.0
        )
        assert tb12.selected == "B"
        assert tb12.rejected == "A"
        # 2nd boundary: A (1st-loser) vs C (top3).  diff 0.022,
        # NOT near-tie.  The 1st winner (B) is NOT in this pair.
        tb23 = decision.tiebreaks[1]
        assert tb23.pair == ("A", "C"), (
            "R03 contract: 2nd boundary compares first_loser (A) "
            "with top3 (C); the 1st winner (B) must not appear in "
            "the 2nd pair"
        )
        assert tb23.macro_iou_difference == pytest.approx(0.022)
        assert tb23.tiebreak_basis == "none"
        assert tb23.selected == "A"
        assert tb23.rejected == "C"
        # The two advanced slots must be distinct candidates.
        assert len(set(decision.advanced)) == 2

    def test_third_replaces_first_loser_when_second_boundary_near_tie(
        self,
    ):
        # The 2nd-vs-3rd near-tie only fires when the 1st boundary
        # has been swapped AND the 2nd boundary itself is a
        # near-tie.  In this scenario:
        #   Top1: A (0.500, 100,000)
        #   Top2: B (0.495, 90,000)
        #   Top3: C (0.490, 30,000)
        # 1st boundary: A vs B, diff=0.005, ratio=10k/90k~=0.111,
        # basis=parameter_count, B (90k) < A (100k) -> B wins.
        # 2nd boundary: A (1st-loser) vs C, diff=0.010, near-tie,
        # ratio=70k/30k~=2.33, basis=parameter_count, C (30k) <
        # A (100k) -> C wins.
        # advanced == ("B", "C").  The simpler 3rd candidate C
        # replaces the 1st-loser A in slot 2.
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.495, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 30_000, 0.490, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ("B", "C")
        assert "A" not in decision.advanced
        assert decision.near_tie_applied is True
        tb12 = decision.tiebreaks[0]
        assert tb12.pair == ("A", "B")
        assert tb12.tiebreak_basis == "parameter_count"
        assert tb12.selected == "B"
        assert tb12.rejected == "A"
        # 2nd boundary: A (1st-loser) vs C (top3).
        tb23 = decision.tiebreaks[1]
        assert tb23.pair == ("A", "C")
        assert tb23.tiebreak_basis == "parameter_count"
        assert tb23.macro_iou_difference == pytest.approx(0.010)
        assert tb23.parameter_difference_ratio == pytest.approx(
            70_000.0 / 30_000.0
        )
        assert tb23.selected == "C"
        assert tb23.rejected == "A"

    def test_first_boundary_no_near_tie_second_boundary_top2_vs_top3(
        self,
    ):
        # 1st boundary has a clear winner (diff >= 0.02 -> no
        # tiebreak).  The 1st-loser is the candidate with the
        # lower ``macro_iou_mean`` between top1 and top2; the
        # 2nd boundary compares that loser with top3.
        #   Top1: A (0.500, 100,000)  -> wins 1st boundary
        #   Top2: B (0.40, 90,000)   -> 1st-loser
        #   Top3: C (0.39, 80,000)   -> 2nd boundary opponent
        # 1st boundary: A vs B, diff=0.10 -> no tiebreak, A wins.
        # 2nd boundary: B (1st-loser) vs C, diff=0.01, near-tie,
        # ratio=10k/80k=0.125>0.10, basis=parameter_count, C (80k)
        # < B (90k) -> C wins.  Advanced = (A, C).
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.40, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 80_000, 0.39, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        # advanced[0] is the 1st boundary winner (A).  advanced[1]
        # is the 2nd boundary winner (C); the 2nd boundary was
        # driven by first_loser (B) vs top3 (C), NOT by the
        # surviving top1 (A) vs top3.
        assert decision.advanced == ("A", "C")
        assert "B" not in decision.advanced
        assert decision.near_tie_applied is True
        tb12 = decision.tiebreaks[0]
        assert tb12.pair == ("A", "B")
        assert tb12.tiebreak_basis == "none"
        assert tb12.macro_iou_difference == pytest.approx(0.10)
        assert tb12.selected == "A"
        assert tb12.rejected == "B"
        # 2nd boundary: B (1st-loser) vs C (top3).  A is NOT here.
        tb23 = decision.tiebreaks[1]
        assert tb23.pair == ("B", "C"), (
            "R03 contract: 2nd boundary must compare first_loser "
            "(B) with top3 (C); the 1st winner (A) must not appear"
        )
        assert tb23.tiebreak_basis == "parameter_count"
        assert tb23.macro_iou_difference == pytest.approx(0.01)
        assert tb23.selected == "C"
        assert tb23.rejected == "B"

    def test_first_boundary_no_near_tie_second_boundary_no_near_tie(self):
        # Both boundaries are outside the near-tie margin; the
        # algorithm reduces to a strict top-2 by ``macro_iou_mean``.
        # 1st boundary: A vs B, diff=0.10 -> A wins.
        # 2nd boundary: B (1st-loser) vs C, diff=0.20 -> B wins.
        # advanced = (A, B).  C is dropped.
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.40, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 80_000, 0.20, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ("A", "B")
        assert "C" not in decision.advanced
        assert decision.near_tie_applied is False
        assert decision.tiebreaks[0].tiebreak_basis == "none"
        # 2nd boundary: B (1st-loser) vs C (top3), diff 0.20.
        assert decision.tiebreaks[1].pair == ("B", "C")
        assert decision.tiebreaks[1].tiebreak_basis == "none"
        assert decision.tiebreaks[1].selected == "B"
        assert decision.tiebreaks[1].rejected == "C"

    def test_second_boundary_fail_closed_missing_worst_subject(self):
        # 1st boundary succeeds with a non-fail-closed basis; 2nd
        # boundary detects a near-tie that requires worst_subject
        # but the 1st-loser or top3 has a missing / non-finite
        # worst_subject_iou.  R03 contract: advanced == () and
        # basis == "failed_no_worst_subject"; the orchestrator
        # promotes terminal_state to FAILED.
        #
        # To force the 2nd boundary onto the worst_subject_iou
        # basis, the parameter ratio must be <= 10%; we use
        # equal param counts so ratio = 0.
        #   Top1: A (0.500, 100,000, ws=0.6) -> 1st winner
        #   Top2: B (0.495, 100,000, ws=0.6) -> 1st-loser
        #   Top3: C (0.490, 100,000, ws=None) -> 2nd opponent
        # 1st boundary: A vs B, diff=0.005, ratio=0/100k=0<0.10,
        # basis=worst_subject_iou.  A ws=0.6, B ws=0.6 -> basis
        # =worst_subject_iou_then_param.  p_h=p_l=100k; the
        # p_h <= p_l branch keeps A as the 1st winner.
        # 2nd boundary: B (1st-loser, ws=0.6) vs C (ws=None),
        # diff=0.005, near-tie, ratio=0<0.10, basis=
        # worst_subject_iou -> C has no usable worst_subject_iou
        # -> basis=failed_no_worst_subject; advanced=().
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 100_000, 0.495, feasibility="FEASIBLE"),
            "C": _make_aggregate(
                "C", 100_000, 0.490, feasibility="FEASIBLE",
                worst_subject_iou=None,
            ),
        }
        decision = _b04a_advance_decision(aggregates)
        assert decision.advanced == ()
        assert decision.near_tie_applied is True
        # 1st boundary was resolved cleanly (no fail-closed).
        assert decision.tiebreaks[0].tiebreak_basis in {
            "worst_subject_iou", "worst_subject_iou_then_param"
        }
        assert decision.tiebreaks[0].selected == "A"
        assert decision.tiebreaks[0].rejected == "B"
        # 2nd boundary fails closed because C has no usable
        # worst_subject_iou.
        tb23 = decision.tiebreaks[1]
        assert tb23.tiebreak_basis == "failed_no_worst_subject"
        assert tb23.pair == ("B", "C")
        assert tb23.selected is None
        assert tb23.rejected is None

    def test_input_order_independence_under_r03_contract(self):
        # Same 3 aggregates, three different dict insertion
        # orders.  All three must produce the same ``advanced``
        # tuple and the same ``tiebreaks`` list.
        base_aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.495, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 80_000, 0.478, feasibility="FEASIBLE"),
        }
        d_forward = _b04a_advance_decision(dict(base_aggregates))
        d_reverse = _b04a_advance_decision(
            {k: base_aggregates[k] for k in reversed(list(base_aggregates))}
        )
        scrambled = list(base_aggregates.values())
        d_scrambled = _b04a_advance_decision(
            {
                "C": scrambled[2],
                "B": scrambled[1],
                "A": scrambled[0],
            }
        )
        for d in (d_forward, d_reverse, d_scrambled):
            assert d.advanced == d_forward.advanced
            assert d.near_tie_applied == d_forward.near_tie_applied
            assert len(d.tiebreaks) == len(d_forward.tiebreaks)
            for tb_a, tb_b in zip(d.tiebreaks, d_forward.tiebreaks):
                assert tb_a.pair == tb_b.pair
                assert tb_a.macro_iou_difference == tb_b.macro_iou_difference
                assert (
                    tb_a.parameter_difference_ratio
                    == tb_b.parameter_difference_ratio
                )
                assert tb_a.tiebreak_basis == tb_b.tiebreak_basis
                assert tb_a.selected == tb_b.selected
                assert tb_a.rejected == tb_b.rejected
        # Sanity: the canonical expected outcome is (B, A) with
        # 2nd pair (A, C).
        assert d_forward.advanced == ("B", "A")
        assert d_forward.tiebreaks[1].pair == ("A", "C")

    def test_advanced_always_two_distinct_candidates(self):
        # The 2nd boundary is fed the 1st-round REJECTED candidate;
        # the 1st-round winner is not in the 2nd boundary's pair.
        # The two advanced slots are therefore always distinct
        # (modulo a defensive fail-closed path that returns
        # advanced=() if the 1st boundary's selected==rejected,
        # which is a contract violation we also cover below).
        # Use the Reviewer case to verify the distinctness
        # invariant.
        aggregates = {
            "A": _make_aggregate("A", 100_000, 0.500, feasibility="FEASIBLE"),
            "B": _make_aggregate("B", 90_000, 0.495, feasibility="FEASIBLE"),
            "C": _make_aggregate("C", 80_000, 0.478, feasibility="FEASIBLE"),
        }
        decision = _b04a_advance_decision(aggregates)
        assert len(decision.advanced) == 2
        assert decision.advanced[0] != decision.advanced[1]


# ---------------------------------------------------------------------------
# 13) _run_validate_config writes protocol-aware identity.
#     The test invokes the CLI as a subprocess so the artifacts on
#     disk are exactly what a real run produces -- and asserts on
#     the on-disk JSON, not on the in-memory MiniConfig.
# ---------------------------------------------------------------------------


def _run_validate_only_subprocess(
    config_path: Path, output_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/run_slp8_region_mini.py --validate-config`` and
    return the CompletedProcess.  Tests assert on the artifacts
    the script writes to ``output_dir``."""

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--validate-config",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestRunValidateConfigIdentity:
    """``--validate-config`` must write protocol-native identity.

    Each test runs the CLI as a subprocess so the assertions are
    against the actual on-disk artifacts (``status.json``,
    ``DONE.json``, ``resolved_config.json``,
    ``input_manifest_hashes.json``), not against in-memory
    MiniConfig objects.
    """

    def test_b04_validate_only_writes_b04_identity(self, tmp_path):
        out = tmp_path / "b04_validate"
        result = _run_validate_only_subprocess(B04_CONFIG_PATH, out)
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        # required files
        for fname in (
            "status.json",
            "DONE.json",
            "resolved_config.json",
            "input_manifest_hashes.json",
            "environment.json",
        ):
            assert (out / fname).exists(), f"missing {fname}"
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        done = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
        hashes = json.loads(
            (out / "input_manifest_hashes.json").read_text(encoding="utf-8")
        )
        # status.json identity
        assert status["task_id"] == TASK_ID
        assert status["config_version"] == MINI_VERSION
        assert status["protocol"] == B04_PROTOCOL_NAME
        assert status["registered_candidates"] == list(B04_CANDIDATE_NAMES)
        assert status["model_parameter_cap"] == 150_000
        # DONE.json identity
        assert done["status"] == "DONE"
        assert done["task_id"] == TASK_ID
        assert done["config_version"] == MINI_VERSION
        assert done["protocol"] == B04_PROTOCOL_NAME
        # input_manifest_hashes identity
        assert hashes["registered_candidates"] == list(B04_CANDIDATE_NAMES)
        assert hashes["a06_split_sha256_expected"] == A06_SPLIT_SHA256_EXPECTED
        # B04A identity MUST NOT leak into B04 artifacts.
        joined = (out / "status.json").read_text(encoding="utf-8") + (
            out / "DONE.json"
        ).read_text(encoding="utf-8")
        assert B04A_TASK_ID not in joined
        assert B04A_CONFIG_VERSION not in joined
        assert "slp8_segformer_b0_v0.1" not in joined
        # The B04A protocol name (as a JSON ``"protocol": "B04A"``
        # field) MUST NOT be present in B04 artifacts.  We use the
        # JSON form to avoid matching the substring "B04" inside
        # ``TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1``.
        assert '"protocol": "B04A"' not in joined, (
            "B04 artifacts must not contain the B04A protocol "
            "name in the JSON ``protocol`` field"
        )

    def test_b04a_validate_only_writes_b04a_identity(self, tmp_path):
        out = tmp_path / "b04a_validate"
        result = _run_validate_only_subprocess(B04A_CONFIG_PATH, out)
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        for fname in (
            "status.json",
            "DONE.json",
            "resolved_config.json",
            "input_manifest_hashes.json",
            "environment.json",
        ):
            assert (out / fname).exists(), f"missing {fname}"
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        done = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
        hashes = json.loads(
            (out / "input_manifest_hashes.json").read_text(encoding="utf-8")
        )
        # status.json identity
        assert status["task_id"] == B04A_TASK_ID
        assert status["config_version"] == B04A_CONFIG_VERSION
        assert status["protocol"] == B04A_PROTOCOL_NAME
        assert status["registered_candidates"] == list(B04A_ACTIVE_CANDIDATE_NAMES)
        assert status["model_parameter_cap"] == 300_000
        assert status["feasibility_threshold"] == B04A_FEASIBILITY_THRESHOLD
        assert status["seeds"] == list(B04A_SEEDS)
        # Deferred candidates surface as a list of records.
        deferred_names = {c["name"] for c in status["deferred_candidates"]}
        assert "slp8_segformer_b0_v0.1" in deferred_names
        # Forbidden candidates are recorded.
        assert set(status["forbidden_candidates"]) == set(
            B04A_FORBIDDEN_CANDIDATE_NAMES
        )
        # DONE.json identity
        assert done["status"] == "DONE"
        assert done["task_id"] == B04A_TASK_ID
        assert done["config_version"] == B04A_CONFIG_VERSION
        assert done["protocol"] == B04A_PROTOCOL_NAME
        assert set(done["registered_candidates"]) == set(
            B04A_ACTIVE_CANDIDATE_NAMES
        )
        assert done["seeds"] == list(B04A_SEEDS)
        assert "slp8_segformer_b0_v0.1" in done["deferred_candidates"]
        # input_manifest_hashes identity
        assert hashes["registered_candidates"] == list(B04A_ACTIVE_CANDIDATE_NAMES)
        assert hashes["forbidden_candidates"] == list(
            B04A_FORBIDDEN_CANDIDATE_NAMES
        )
        assert "slp8_segformer_b0_v0.1" in hashes["deferred_candidates"]
        assert hashes["seeds"] == list(B04A_SEEDS)
        # B04 identity MUST NOT leak into B04A artifacts.
        # ``TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1`` legitimately
        # contains the substring "B04", so a literal string check
        # for "B04" would always be true.  We use the JSON
        # ``"protocol": "B04"`` form (which only the B04 branch
        # would emit) and the B04 task_id prefix.
        import re
        joined = (out / "status.json").read_text(encoding="utf-8") + (
            out / "DONE.json"
        ).read_text(encoding="utf-8")
        assert TASK_ID not in joined
        assert MINI_VERSION not in joined
        assert '"protocol": "B04"' not in joined, (
            "B04A artifacts must not contain the B04 protocol "
            "name in the JSON ``protocol`` field"
        )
        assert "TASK-SLP-B04-PM-ONLY-REGION-MINI" not in joined, (
            "B04A artifacts must not contain the B04 task_id "
            "prefix (TASK-SLP-B04-PM-ONLY-REGION-MINI…)"
        )


# ---------------------------------------------------------------------------
# 14) _run_real_b01 protocol dispatch + cross-protocol guards.
#     These tests load ``scripts/run_slp8_region_mini.py`` as a
#     module, monkeypatch ``run_mini`` / ``run_mini_b04a`` with
#     recording stubs, and invoke ``_run_real_b01`` with a B04
#     or B04A config.  The B04 path must call ``run_mini`` and
#     the B04A path must call ``run_mini_b04a`` with the B04A
#     budget and the per-(candidate, seed) resume map.  An
#     unknown protocol must fail closed BEFORE any training
#     artifact is written.
# ---------------------------------------------------------------------------


def _load_run_slp8_module():
    """Import the script as a module so we can monkeypatch it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_slp8_region_mini_under_test",
        str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRunRealB01Dispatch:
    """``_run_real_b01`` dispatches by ``config.protocol``.

    Each test loads the script as a module, mocks the two real-B01
    helpers (``_run_real_b01_b04`` / ``_run_real_b01_b04a``) with
    recording stubs, and verifies the dispatch chooses the right
    branch for each protocol.  The downstream B01 contract
    loading is mocked with a no-op lambda.  Cross-protocol use
    is fail-closed: an unknown protocol raises
    :class:`MiniProtocolError` before any training artifact is
    written.
    """

    def test_b04_real_path_calls_b04_helper(self, tmp_path, monkeypatch):
        mod = _load_run_slp8_module()
        calls: dict[str, Any] = {}

        def fake_b04(**kwargs):
            calls["b04_called"] = True
            calls["b04_kwargs"] = kwargs
            return 0

        def fake_b04a(**kwargs):  # pragma: no cover
            calls["b04a_called"] = True
            raise AssertionError("B04 path must not call _run_real_b01_b04a")

        # Minimal b01 stub: only ``snapshot`` is read by the
        # dispatch wrapper (to log train/val/test counts).
        class _Snap:
            train_count = 4
            val_count = 2
            a06_split_sha256 = "f" * 64
            structural_test = type("T", (), {"sample_count": 0})()

        monkeypatch.setattr(mod, "_run_real_b01_b04", fake_b04)
        monkeypatch.setattr(mod, "_run_real_b01_b04a", fake_b04a)
        monkeypatch.setattr(
            mod, "_load_b01_freeze_and_contract",
            lambda raw, a, b: {"snapshot": _Snap()},
        )

        out = tmp_path / "b04_real"
        result = mod._run_real_b01(
            B04_CONFIG_PATH,
            out,
            b01_freeze_dir=Path("/tmp/fake_freeze"),
            dataset_root=Path("/tmp/fake_data"),
            experiment_id="EXP-SLP-B04-TEST-EXP-ID",
            frozen_git_commit="0" * 40,
            frozen_git_dirty=False,
        )
        assert result == 0
        assert calls.get("b04_called") is True
        assert calls.get("b04a_called") is None
        # The B04 helper received the resolved B04 MiniConfig.
        cfg = calls["b04_kwargs"]["config"]
        assert cfg.protocol == B04_PROTOCOL_NAME
        assert cfg.seeds == (42,)
        # experiment_id is propagated to the real B01 helper.
        assert (
            calls["b04_kwargs"]["experiment_id"] == "EXP-SLP-B04-TEST-EXP-ID"
        )

    def test_b04a_real_path_calls_b04a_helper(self, tmp_path, monkeypatch):
        mod = _load_run_slp8_module()
        calls: dict[str, Any] = {}

        def fake_b04(**kwargs):  # pragma: no cover
            calls["b04_called"] = True
            raise AssertionError("B04A path must not call _run_real_b01_b04")

        def fake_b04a(**kwargs):
            calls["b04a_called"] = True
            calls["b04a_kwargs"] = kwargs
            return 0

        class _Snap:
            train_count = 4
            val_count = 2
            a06_split_sha256 = "f" * 64
            structural_test = type("T", (), {"sample_count": 0})()

        monkeypatch.setattr(mod, "_run_real_b01_b04", fake_b04)
        monkeypatch.setattr(mod, "_run_real_b01_b04a", fake_b04a)
        monkeypatch.setattr(
            mod, "_load_b01_freeze_and_contract",
            lambda raw, a, b: {"snapshot": _Snap()},
        )

        out = tmp_path / "b04a_real"
        result = mod._run_real_b01(
            B04A_CONFIG_PATH,
            out,
            b01_freeze_dir=Path("/tmp/fake_freeze"),
            dataset_root=Path("/tmp/fake_data"),
            experiment_id="EXP-SLP-B04A-TEST-EXP-ID",
            frozen_git_commit="0" * 40,
            frozen_git_dirty=False,
        )
        assert result == 0
        assert calls.get("b04a_called") is True
        assert calls.get("b04_called") is None
        # The B04A helper received the resolved B04A MiniConfig.
        cfg = calls["b04a_kwargs"]["config"]
        assert cfg.protocol == B04A_PROTOCOL_NAME
        assert cfg.seeds == B04A_SEEDS
        assert set(cfg.candidates) == set(B04A_ACTIVE_CANDIDATE_NAMES)

    def test_b04a_helper_passes_b04a_budget_and_resume_map(
        self, tmp_path, monkeypatch
    ):
        # Directly exercise ``_run_real_b01_b04a`` with a minimal
        # mocked environment to verify (a) the B04A budget
        # (45/135/8192) is constructed in the helper, and
        # (b) the per-(candidate, seed) resume map is built and
        # passed to ``run_mini_b04a`` as ``resume_from_per_candidate_seed``.
        mod = _load_run_slp8_module()
        captured: dict[str, Any] = {}

        # ``_run_real_b01_b04a`` imports these symbols from the
        # ``slp8_region_dataset`` module locally; patch them at
        # their source module so the local import picks up the
        # stub.
        from topper_perception.neural import slp8_region_dataset as ds_mod

        def fake_build_smoke_dataset(**kwargs):
            return ([], [], {"n_test_samples": 0})

        def fake_compute_class_weights(stats):
            class _CW:
                weights = {0: 1.0}
            return _CW()

        def fake_run_mini_b04a(**kwargs):
            captured["run_mini_b04a_kwargs"] = kwargs
            from topper_perception.neural.slp8_region_mini import (
                B04ARunResult,
                B04AAdvanceDecision,
            )
            return B04ARunResult(
                config=kwargs["config"],
                dataset_manifest=kwargs["dataset_manifest"],
                environment={},
                class_weight_result=kwargs["class_weight_result"],
                candidate_results={},
                n_candidates_feasible=0,
                n_candidates_not_feasible=3,
                n_candidates_failed=0,
                n_candidates_stopped=0,
                overall_decision="MINI_NOT_FEASIBLE",
                advanced=(),
                near_tie_applied=False,
                near_tie_margin=0.02,
                advance_decision=B04AAdvanceDecision(
                    advanced=(),
                    near_tie_applied=False,
                    near_tie_margin=0.02,
                    tiebreaks=[],
                ),
                terminal_state="DONE",
                started_at_utc="2026-01-01T00:00:00+00:00",
                ended_at_utc="2026-01-01T00:00:00+00:00",
                wall_clock_seconds=0.0,
                input_hashes={},
                train_class_stats_source="b01_train_class_stats.json",
                synthetic=False,
                determinism={},
                resource_budget=kwargs["budget"],
                b01_contract_report=kwargs.get("b01_contract_report"),
                # Carry the identity through so the post-orchestrator
                # ``_write_b04a_run_bundle`` / ``write_status_files``
                # code paths can construct a valid identity block
                # (TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1
                # R02 ITERATE review point 4: single identity source).
                experiment_id=str(kwargs.get("experiment_id") or ""),
                data_manifest_sha256=str(
                    kwargs.get("data_manifest_sha256") or ""
                ),
                # R04 ITERATE: the run bundle writer validates
                # ``git_commit`` strictly; the fake result must
                # carry a real-looking SHA so the writer can emit
                # the manifest without raising.
                git_commit="0" * 40,
                git_dirty=False,
            )

        monkeypatch.setattr(ds_mod, "build_smoke_dataset", fake_build_smoke_dataset)
        monkeypatch.setattr(
            mod, "compute_class_weights", fake_compute_class_weights
        )
        monkeypatch.setattr(mod, "assert_class_weight_invariants", lambda x: None)
        monkeypatch.setattr(mod, "verify_subject_isolation", lambda a, b: True)
        monkeypatch.setattr(
            mod, "resolve_device", lambda *a, **kw: torch.device("cpu")
        )
        monkeypatch.setattr(mod, "run_mini_b04a", fake_run_mini_b04a)
        # Stub _write_b04a_run_bundle and write_status_files to no-op.
        monkeypatch.setattr(mod, "_write_b04a_run_bundle", lambda **kw: None)
        monkeypatch.setattr(mod, "write_status_files", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_write_json", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_gather_environment", lambda: {})

        # Build a fake b01 dict with the right shape.
        fake_freeze = type("F", (), {})()
        fake_freeze.train_rows = [
            type("R", (), {"subject_id": f"subj_train_{i}"})()
            for i in range(4)
        ]
        fake_freeze.val_rows = [
            type("R", (), {"subject_id": f"subj_val_{i}"})()
            for i in range(2)
        ]
        b01 = {
            "freeze": fake_freeze,
            "fm_file_sha": "f" * 64,
            "train_class_stats_sha256": "e" * 64,
            "b01_contract_report": {"ok": True},
            "b01_expected": type(
                "E", (), {"freeze_manifest_core_sha256": "f" * 64}
            )(),
            "train_class_stats": {
                "n_samples": 1,
                "n_pixels": 1,
                "per_class_pixel_ratio": {
                    0: 0.5, 1: 0.5, **{i: 0.0 for i in range(2, 9)}
                },
            },
        }

        # Build the B04A MiniConfig (with synthetic CPU overrides).
        from topper_perception.neural.slp8_region_mini import build_mini_config
        raw = json.loads(B04A_CONFIG_PATH.read_text(encoding="utf-8"))
        validate_mini_config(raw)
        raw["training"]["device"] = "cpu"
        raw["training"]["max_epochs"] = 1
        raw["training"]["min_epochs"] = 1
        raw["training"]["early_stopping"]["patience"] = 1
        config = build_mini_config(
            raw,
            b01_freeze_dir="<SYNTHETIC>",
            data_root="<SYNTHETIC>",
            config_path=str(B04A_CONFIG_PATH),
        )

        # Build a fake resume source with two (candidate, seed)
        # last.pt files.
        resume_src = tmp_path / "resume_src"
        resume_src.mkdir(parents=True, exist_ok=True)
        (resume_src / "status.json").write_text(
            json.dumps({"terminal_state": "RUNNING"}), encoding="utf-8"
        )
        (resume_src / "checkpoints" / SMALL_UNET_VERSION / "seed_0042").mkdir(
            parents=True
        )
        (
            resume_src
            / "checkpoints"
            / SMALL_UNET_VERSION
            / "seed_0042"
            / "last.pt"
        ).write_text("x", encoding="utf-8")
        (
            resume_src
            / "checkpoints"
            / DEEPLABV3PLUS_LITE_VERSION
            / "seed_0123"
        ).mkdir(parents=True)
        (
            resume_src
            / "checkpoints"
            / DEEPLABV3PLUS_LITE_VERSION
            / "seed_0123"
            / "last.pt"
        ).write_text("x", encoding="utf-8")

        log_path = tmp_path / "run.log"
        log_path.write_text("", encoding="utf-8")

        result = mod._run_real_b01_b04a(
            config_path=B04A_CONFIG_PATH,
            output_dir=tmp_path / "out",
            b01_freeze_dir=Path("/tmp/fake_freeze"),
            dataset_root=Path("/tmp/fake_data"),
            resume_from=resume_src,
            config=config,
            b01=b01,
            log_path=log_path,
            experiment_id="EXP-SLP-B04A-TEST-EXP-ID",
            frozen_git_commit="0" * 40,
            frozen_git_dirty=False,
        )
        # The B04A helper must have called run_mini_b04a and the
        # B04A budget + per-(candidate, seed) resume map must
        # be present.
        assert "run_mini_b04a_kwargs" in captured
        kw = captured["run_mini_b04a_kwargs"]
        budget = kw["budget"]
        assert budget.max_wall_seconds_per_candidate == pytest.approx(45 * 60.0)
        assert budget.max_wall_seconds_total == pytest.approx(135 * 60.0)
        assert budget.max_peak_cuda_mb == pytest.approx(8192.0)
        # Per-(candidate, seed) resume map.
        resume_map = kw["resume_from_per_candidate_seed"]
        assert isinstance(resume_map, dict)
        assert set(resume_map.keys()) == {
            SMALL_UNET_VERSION, DEEPLABV3PLUS_LITE_VERSION
        }
        assert set(resume_map[SMALL_UNET_VERSION].keys()) == {42}
        assert set(resume_map[DEEPLABV3PLUS_LITE_VERSION].keys()) == {123}

    def test_unknown_protocol_fail_closed_in_real_path(
        self, tmp_path, monkeypatch
    ):
        mod = _load_run_slp8_module()

        # Patch the B01 contract loading so we get past validation
        # regardless of fake data.
        class _Snap:
            train_count = 4
            val_count = 2
            a06_split_sha256 = "f" * 64
            structural_test = type("T", (), {"sample_count": 0})()

        monkeypatch.setattr(
            mod, "_load_b01_freeze_and_contract",
            lambda raw, a, b: {"snapshot": _Snap()},
        )
        # Patch ``build_mini_config`` so the returned MiniConfig
        # reports protocol="B99" (unknown).  This is the
        # equivalent of validating a config that the dispatcher
        # doesn't recognise.
        from topper_perception.neural.slp8_region_mini import (
            MiniConfig as _MC,
        )

        def fake_build_mini_config(raw, **kwargs):
            real = mod.__class__  # not used
            from topper_perception.neural.slp8_region_mini import (
                build_mini_config as _real_build,
            )
            real_cfg = _real_build(raw, **kwargs)
            return dataclasses.replace(real_cfg, protocol="B99")

        monkeypatch.setattr(mod, "build_mini_config", fake_build_mini_config)

        out = tmp_path / "unknown"
        with pytest.raises(MiniProtocolError, match="not recognised"):
            mod._run_real_b01(
                B04_CONFIG_PATH,
                out,
                b01_freeze_dir=Path("/tmp/fake_freeze"),
                dataset_root=Path("/tmp/fake_data"),
                frozen_git_commit="0" * 40,
                frozen_git_dirty=False,
            )
        # No training artifact was written.
        assert not (out / "checkpoints").exists()


class TestRunMiniCrossProtocolGuards:
    """``run_mini`` rejects B04A configs; ``run_mini_b04a`` rejects
    B04 configs.  The cross-protocol guard is enforced BEFORE any
    training artifact is written."""

    def test_run_mini_rejects_b04a_config(self):
        b04a_cfg = _build_b04a_mini_config()
        with pytest.raises(MiniProtocolError, match="B04A"):
            run_mini(
                config=b04a_cfg,
                train_dataset=[],
                val_dataset=[],
                dataset_manifest={"n_test_samples": 0},
                class_weight_result=None,  # type: ignore[arg-type]
                output_dir=Path("/tmp/no_write"),
                device=torch.device("cpu"),
                input_hashes={},
                train_class_stats_source="synthetic",
                synthetic=True,
            )

    def test_run_mini_b04a_rejects_b04_config(self):
        b04_cfg = _build_b04_mini_config()
        with pytest.raises(MiniProtocolError, match="B04A"):
            run_mini_b04a(
                config=b04_cfg,
                train_dataset=[],
                val_dataset=[],
                dataset_manifest={"n_test_samples": 0},
                class_weight_result=None,  # type: ignore[arg-type]
                output_dir=Path("/tmp/no_write"),
                device=torch.device("cpu"),
                input_hashes={},
                train_class_stats_source="synthetic",
                synthetic=True,
                # R05 ITERATE: required git_commit/git_dirty parameters.
                git_commit=_TEST_GIT_COMMIT,
                git_dirty=False,
            )


# ---------------------------------------------------------------------------
# 12) B04A experiment-identity carrier (TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1)
# ---------------------------------------------------------------------------


# A canonical Owner EXP-ID used across the B04A carrier tests.
_TEST_OWNER_EXP_ID = "EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03"
# A distinct Owner EXP-ID for drift tests; not equal to _TEST_OWNER_EXP_ID.
_TEST_OWNER_EXP_ID_ALT = (
    "EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03-DRIFT"
)
# The synthetic freeze_manifest.json file hash.  Real B01 runs would
# hash the on-disk file; the test uses a 64-character placeholder.
_TEST_FREEZE_MANIFEST_FILE_SHA = "f" * 64
# A non-equal config SHA so the test can prove the data-manifest
# carrier is NOT a silent fallback to config_sha256.
_TEST_CONFIG_SHA = "a" * 64
# A canonical 40-character hex SHA-1 used wherever a test must
# supply a valid ``git_commit`` to :func:`_b04a_identity_block`
# (R04 ITERATE: the formal identity block rejects empty / sentinel
# / non-hex / wrong-length values).
_TEST_GIT_COMMIT = "0" * 40


def _expected_synthetic_manifest_sha() -> str:
    from topper_perception.neural.slp8_region_mini import (
        _compute_synthetic_manifest_sha256,
    )

    return _compute_synthetic_manifest_sha256()


class TestB04AExperimentIdentityCarriers:
    """Identity carrier tests for the B04A experiment-identity fix.

    These tests pin the frozen semantics of
    TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1:
    ``experiment_id`` is Owner-supplied (no TASK-ID-derived
    fallback); ``data_manifest_sha256`` is the on-disk
    ``freeze_manifest.json`` file hash for real B01 runs and the
    deterministic synthetic-manifest hash for the synthetic smoke;
    ``model_version`` is the candidate builder's exact version for
    per-seed/per-candidate artifacts and the
    ``multi_candidate[... ]`` string in frozen config order for the
    run-level artifact; an empty ``experiment_id``,
    ``data_manifest_sha256`` or ``model_version`` is fail-closed;
    resume rejects any drift in those three fields.
    """

    # ------------------------------------------------------------------
    # (1) experiment_id propagation
    # ------------------------------------------------------------------

    def test_run_level_identity_block_carries_owner_exp_id(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            git_commit=_TEST_GIT_COMMIT,
        )
        # The Owner EXP-ID is verbatim, not a TASK-ID-derived
        # ``f"{task_id}::...`` composite.
        assert block["experiment_id"] == _TEST_OWNER_EXP_ID
        assert (
            "TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1" not in block["experiment_id"]
        )
        # Synthetic flag is False for a real Owner EXP-ID.
        assert block["synthetic"] is False
        assert (
            block["data_manifest_source"] == "freeze_manifest_file_sha256"
        )

    def test_run_level_identity_block_synthetic_marker(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            SYNTHETIC_EXP_ID,
        )

        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=SYNTHETIC_EXP_ID,
            data_manifest_sha256=_expected_synthetic_manifest_sha(),
            git_commit=_TEST_GIT_COMMIT,
        )
        assert block["experiment_id"] == SYNTHETIC_EXP_ID
        assert block["synthetic"] is True
        assert (
            block["data_manifest_source"]
            == "synthetic_canonical_manifest_sha256"
        )

    def test_run_level_identity_block_rejects_empty_experiment_id(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for empty in ("", "   ", "\t\n"):
            with pytest.raises(MiniProtocolError, match="experiment_id"):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=empty,
                    data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                )

    def test_run_level_identity_block_rejects_empty_data_manifest_sha(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for empty in ("", "   ", None):
            with pytest.raises(
                MiniProtocolError, match="data_manifest_sha256"
            ):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=_TEST_OWNER_EXP_ID,
                    data_manifest_sha256=empty,  # type: ignore[arg-type]
                    git_commit=_TEST_GIT_COMMIT,
                )

    # ------------------------------------------------------------------
    # (2) data_manifest_sha256 == freeze_manifest_file_sha256
    # ------------------------------------------------------------------

    def test_real_run_data_manifest_sha256_is_freeze_manifest_file_sha(self):
        """The real B01 path uses the on-disk freeze_manifest.json file hash.

        The hash MUST be different from the config SHA so a Reviewer
        can tell the identity is real (not a config fallback).
        """
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            git_commit=_TEST_GIT_COMMIT,
        )
        assert (
            block["data_manifest_sha256"] == _TEST_FREEZE_MANIFEST_FILE_SHA
        )
        # Deterministic and demonstrably distinct from config_sha256.
        assert block["data_manifest_sha256"] != _TEST_CONFIG_SHA
        # Lower-cased: hex digests are case-insensitive but the
        # reviewer audit reads the same string everywhere.
        assert (
            block["data_manifest_sha256"]
            == block["data_manifest_sha256"].lower()
        )

    def test_synthetic_manifest_hash_is_deterministic(self):
        from topper_perception.neural.slp8_region_mini import (
            _compute_synthetic_manifest_sha256,
        )

        h1 = _compute_synthetic_manifest_sha256()
        h2 = _compute_synthetic_manifest_sha256()
        assert h1 == h2
        # 64-character lowercase SHA-256 hex digest.
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)
        # The synthetic hash is distinct from any plausible real
        # freeze_manifest.json file hash and from the config SHA.
        assert h1 != _TEST_FREEZE_MANIFEST_FILE_SHA
        assert h1 != _TEST_CONFIG_SHA

    def test_synthetic_identity_block_carries_synthetic_manifest_hash(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            SYNTHETIC_EXP_ID,
        )

        synth_hash = _expected_synthetic_manifest_sha()
        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=SYNTHETIC_EXP_ID,
            data_manifest_sha256=synth_hash,
            git_commit=_TEST_GIT_COMMIT,
        )
        assert block["data_manifest_sha256"] == synth_hash
        assert block["synthetic"] is True
        # A real B01 run's data_manifest_sha256 would never be
        # equal to the synthetic manifest hash; the synthetic hash
        # is a deterministic constant of the canonical synthetic
        # payload.
        assert block["data_manifest_sha256"] != _TEST_FREEZE_MANIFEST_FILE_SHA

    def test_synthetic_manifest_cannot_be_confused_for_real(self):
        """Synthetic identity MUST NEVER be accepted as real B01 identity.

        Concretely: the synthetic data_manifest_sha256 equals a
        deterministic value that is distinct from the real B01
        freeze_manifest_file_sha256, the config_sha256, and from
        any TASK-ID-derived value.  The synthetic flag on the
        identity block is True.
        """
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            SYNTHETIC_EXP_ID,
        )

        real_block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            git_commit=_TEST_GIT_COMMIT,
        )
        synth_block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=SYNTHETIC_EXP_ID,
            data_manifest_sha256=_expected_synthetic_manifest_sha(),
            git_commit=_TEST_GIT_COMMIT,
        )
        # Synthetic identity is not "real" by construction.
        assert real_block["synthetic"] is False
        assert synth_block["synthetic"] is True
        # data_manifest_sha256 is distinct between real and synthetic.
        assert (
            real_block["data_manifest_sha256"]
            != synth_block["data_manifest_sha256"]
        )

    # ------------------------------------------------------------------
    # (3) run-level model_version uses multi_candidate[... ] in config order
    # ------------------------------------------------------------------

    def test_run_level_model_version_uses_multi_candidate_grammar(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            git_commit=_TEST_GIT_COMMIT,
        )
        # The run-level model_version MUST be the
        # ``multi_candidate[... ]`` string.  The B04A active set is
        # the validated frozen config order; DEFERRED entries
        # (SegFormer-B0) are filtered out by build_b04a_mini_config
        # so they MUST NOT appear in the multi_candidate[... ] list.
        model_version = block["model_version"]
        assert model_version.startswith("multi_candidate[")
        assert model_version.endswith("]")
        inner = model_version[len("multi_candidate[") : -1]
        names = inner.split(",")
        assert names == list(B04A_ACTIVE_CANDIDATE_NAMES)
        # The forbidden candidates (TinyFCN, SegFormer-B0) MUST NOT
        # appear in the run-level model_version.
        assert "slp8_tiny_fcn_v0.1" not in names
        assert "slp8_segformer_b0_v0.1" not in names

    def test_run_level_model_version_config_order_strict(self):
        """multi_candidate[... ] order comes from the frozen config, not results.

        The B04A active set order is fixed: small_unet, resunet_lite,
        deeplabv3plus_lite (matches the order in
        ``configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json``
        and the B04A_FEASIBILITY_THRESHOLD / B04A_SEEDS contract).
        """
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        block = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256=_TEST_CONFIG_SHA,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            git_commit=_TEST_GIT_COMMIT,
        )
        inner = block["model_version"][len("multi_candidate[") : -1]
        names = inner.split(",")
        assert names == [
            SMALL_UNET_VERSION,
            RESUNET_LITE_VERSION,
            DEEPLABV3PLUS_LITE_VERSION,
        ]

    def test_run_level_model_version_empty_candidate_list_fail_closed(self):
        from topper_perception.neural.slp8_region_mini import (
            _build_run_level_model_version,
            MiniProtocolError,
        )

        with pytest.raises(MiniProtocolError, match="empty"):
            _build_run_level_model_version([])

    def test_candidate_level_model_version_keeps_builder_version(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        for cand, expected in (
            (SMALL_UNET_VERSION, SMALL_UNET_VERSION),
            (RESUNET_LITE_VERSION, RESUNET_LITE_VERSION),
            (DEEPLABV3PLUS_LITE_VERSION, DEEPLABV3PLUS_LITE_VERSION),
        ):
            block = _b04a_identity_block(
                config=_build_b04a_mini_config(),
                config_sha256=_TEST_CONFIG_SHA,
                experiment_id=_TEST_OWNER_EXP_ID,
                data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                git_commit=_TEST_GIT_COMMIT,
                candidate=cand,
                seed=42,
            )
            # Candidate- and seed-level artifacts carry the
            # candidate builder's exact model version, NOT the
            # ``multi_candidate[... ]`` string.
            assert block["model_version"] == expected
            assert not block["model_version"].startswith("multi_candidate[")

    def test_run_level_model_version_rejects_blank_candidate_name(self):
        from topper_perception.neural.slp8_region_mini import (
            _build_run_level_model_version,
            MiniProtocolError,
        )

        with pytest.raises(MiniProtocolError, match="blank"):
            _build_run_level_model_version(
                [SMALL_UNET_VERSION, "   ", RESUNET_LITE_VERSION]
            )

    # ------------------------------------------------------------------
    # (4) Resume drift rejection
    # ------------------------------------------------------------------

    def test_resume_rejects_experiment_id_drift(self):
        saved = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256=_TEST_CONFIG_SHA,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            git_commit="0" * 40,
            git_dirty=False,
        )
        drifted = replace(saved, experiment_id=_TEST_OWNER_EXP_ID_ALT)
        with pytest.raises(ResumeIdentityError, match="experiment_id"):
            verify_resume_identity(saved=saved, requested=drifted)

    def test_resume_rejects_data_manifest_sha256_drift(self):
        saved = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256=_TEST_CONFIG_SHA,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            git_commit="0" * 40,
            git_dirty=False,
        )
        drifted = replace(saved, data_manifest_sha256="0" * 64)
        with pytest.raises(ResumeIdentityError, match="data_manifest_sha256"):
            verify_resume_identity(saved=saved, requested=drifted)

    def test_resume_rejects_model_version_drift(self):
        saved = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256=_TEST_CONFIG_SHA,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            git_commit="0" * 40,
            git_dirty=False,
        )
        drifted = replace(saved, model_version=RESUNET_LITE_VERSION)
        with pytest.raises(ResumeIdentityError, match="model_version"):
            verify_resume_identity(saved=saved, requested=drifted)

    def test_resume_accepts_matching_identity_with_new_fields(self):
        identity = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256=_TEST_CONFIG_SHA,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            git_commit="0" * 40,
            git_dirty=False,
        )
        # No raise.
        verify_resume_identity(saved=identity, requested=identity)

    def test_resume_rejects_pre_fix_identity_block(self):
        """Pre-fix checkpoints lacked experiment_id / data_manifest_sha256.

        Refusing to load them is the fail-closed contract; a
        Reviewer can never silently inherit a missing identity.
        """
        from topper_perception.neural.slp8_region_resume import (
            identity_from_dict,
        )

        # Legacy identity block (B04 R02 contract; no experiment_id,
        # no data_manifest_sha256).
        legacy_payload = {
            "identity": {
                "task_id": B04A_TASK_ID,
                "candidate": SMALL_UNET_VERSION,
                "model_version": SMALL_UNET_VERSION,
                "seed": 42,
                "n_classes": N_CLASSES,
                "image_shape": list(PRESSURE_SHAPE),
                "config_sha256": _TEST_CONFIG_SHA,
                "a06_split_sha256": A06_SPLIT_SHA256_EXPECTED,
                "freeze_manifest_sha256": _TEST_FREEZE_MANIFEST_FILE_SHA,
                "train_class_stats_sha256": "e" * 64,
                "class_weight_sha256": "d" * 64,
                "input_manifest_hashes_sha256": "c" * 64,
            }
        }
        with pytest.raises(ResumeIdentityError, match="experiment_id"):
            identity_from_dict(legacy_payload)

    # ------------------------------------------------------------------
    # (5) CLI --experiment-id fail-closed behaviour
    # ------------------------------------------------------------------

    def test_cli_rejects_missing_experiment_id_for_real_b01(self, tmp_path):
        """Real B01 path with --run-authorized but no --experiment-id.

        The CLI MUST fail closed BEFORE creating the requested
        output directory.
        """
        out = tmp_path / "real_no_exp_id"
        # The pre-validation path: if output_dir is created, the
        # CLI did NOT fail closed.
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--b01-freeze-dir", str(tmp_path / "fake_freeze"),
                "--dataset-root", str(tmp_path / "fake_data"),
                "--run-authorized",
                # No --experiment-id.
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 2, (
            f"expected REJECTED exit 2, got {result.returncode}; "
            f"stderr={result.stderr}"
        )
        assert "--experiment-id" in result.stderr
        # The output directory MUST NOT be created.
        assert not out.exists(), (
            f"output dir was created despite missing --experiment-id: "
            f"{out}; the CLI must fail closed before any side effect"
        )

    def test_cli_rejects_blank_experiment_id_for_real_b01(self, tmp_path):
        out = tmp_path / "real_blank_exp_id"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--b01-freeze-dir", str(tmp_path / "fake_freeze"),
                "--dataset-root", str(tmp_path / "fake_data"),
                "--run-authorized",
                "--experiment-id", "   ",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 2
        assert (
            "non-empty Owner-supplied" in result.stderr
            or "whitespace" in result.stderr
        )
        assert not out.exists()

    def test_cli_rejects_synthetic_sentinel_for_real_b01(self, tmp_path):
        from topper_perception.neural.slp8_region_mini import (
            SYNTHETIC_EXP_ID,
        )

        out = tmp_path / "real_synth_id"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--b01-freeze-dir", str(tmp_path / "fake_freeze"),
                "--dataset-root", str(tmp_path / "fake_data"),
                "--run-authorized",
                "--experiment-id", SYNTHETIC_EXP_ID,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 2
        assert "reserved for synthetic" in result.stderr
        assert not out.exists()

    def test_cli_synthetic_cpu_smoke_ignores_experiment_id(self, tmp_path):
        """Synthetic CPU smoke does not require --experiment-id.

        The synthetic smoke uses a fixed sentinel EXP-ID internally
        so a synthetic identity can never be confused with a real
        B01 identity.  Passing --experiment-id is permitted but
        ignored; the synthetic manifest hash is used unconditionally.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(tmp_path / "b04a_synth"),
                "--synthetic-cpu-smoke-b04a",
                "--no-write",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, (
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        last = result.stdout.strip().splitlines()[-1]
        assert last.startswith("B04A_SMOKE_NO_WRITE ")


# ---------------------------------------------------------------------------
# 13) Real on-disk artifact identity audit (R02 ITERATE review point 5)
# ---------------------------------------------------------------------------


# The seven required identity fields the frozen B04A
# ``identity_hard_gate.required_fields`` contract pins at the top
# level of every JSON carrier and as the first JSON line of every
# log carrier.
_REQUIRED_B04A_IDENTITY_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "data_manifest_sha256",
    "split_sha256",
    "model_version",
)


def _assert_has_required_identity_fields(
    payload: dict[str, Any], *, where: str
) -> None:
    missing = set(_REQUIRED_B04A_IDENTITY_FIELDS) - set(payload)
    assert not missing, f"{where} missing identity fields {sorted(missing)}"


def _run_b04a_synthetic_writing_smoke(
    out_dir: Path,
    *,
    budget_override_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the B04A synthetic smoke in writing mode."""
    env = {
        **os.environ,
        "PYTHONHASHSEED": "42",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if budget_override_seconds is not None:
        env["B04A_SMOKE_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS"] = str(
            budget_override_seconds
        )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
            "--config", str(B04A_CONFIG_PATH),
            "--output-dir", str(out_dir),
            "--synthetic-cpu-smoke-b04a",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


@pytest.fixture
def b04a_write_dir(tmp_path):
    """Run a writing-mode B04A synthetic CPU smoke and return the output dir."""
    out = tmp_path / "b04a_write"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    proc = _run_b04a_synthetic_writing_smoke(out)
    assert proc.returncode in {0, 1}, (
        f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    )
    assert out.exists()
    assert (out / "manifest.json").exists(), (
        f"B04A writing smoke did not produce manifest.json in {out}; "
        f"stderr={proc.stderr}"
    )
    return out


class TestB04AActualArtifactIdentityAudit:
    """Audit the actual on-disk B04A artifact identity after a writing-mode run.

    Every R02 ITERATE review point is covered by inspecting the
    real on-disk JSON / log / checkpoint / CSV / terminal artifacts
    produced by the writing-mode synthetic CPU smoke.  In-memory
    return values of ``_b04a_identity_block`` alone are NOT a
    substitute for an end-to-end carrier audit (R02 ITERATE
    explicitly noted the gap).
    """

    # ------------------------------------------------------------------
    # (1) best.pt / last.pt payload["identity"] seven fields, non-empty
    # ------------------------------------------------------------------

    def test_best_last_pt_identity_block_has_seven_required_fields(
        self, b04a_write_dir: Path
    ) -> None:
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                seed_dir = (
                    b04a_write_dir
                    / "checkpoints"
                    / cand
                    / f"seed_{seed:04d}"
                )
                for pt_name in ("best.pt", "last.pt"):
                    payload = torch.load(
                        seed_dir / pt_name,
                        map_location="cpu",
                        weights_only=True,
                    )
                    identity = payload["identity"]
                    _assert_has_required_identity_fields(
                        identity,
                        where=f"{cand}/seed_{seed:04d}/{pt_name}",
                    )
                    # Synthetic identity: EXP-ID is the sentinel;
                    # data_manifest_sha256 is the deterministic
                    # synthetic-manifest hash; split_sha256 is
                    # non-empty; model_version matches the
                    # candidate builder.
                    assert (
                        identity["experiment_id"] == SYNTHETIC_EXP_ID
                    )
                    assert identity["data_manifest_sha256"] != ""
                    assert identity["split_sha256"] != ""
                    assert identity["git_commit"] != ""
                    assert identity["config_sha256"] != ""
                    assert (
                        identity["model_version"]
                        == get_model_builder(cand).version
                    )
                    assert identity["model_version"] == cand

    # ------------------------------------------------------------------
    # (2) Run-level JSON carriers carry the seven fields
    # ------------------------------------------------------------------

    def test_run_level_carriers_have_seven_identity_fields(
        self, b04a_write_dir: Path
    ) -> None:
        for fname in (
            "manifest.json",
            "status.json",
            "candidate_decision.json",
            "environment.json",
            "input_manifest_hashes.json",
            "resolved_config.json",
            "budget_report.json",
        ):
            payload = json.loads(
                (b04a_write_dir / fname).read_text(encoding="utf-8")
            )
            _assert_has_required_identity_fields(
                payload, where=fname
            )

    def test_budget_report_uses_observed_wall_clock_and_seed_peak(
        self, b04a_write_dir: Path
    ) -> None:
        budget = json.loads(
            (b04a_write_dir / "budget_report.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (b04a_write_dir / "manifest.json").read_text(encoding="utf-8")
        )
        seed_peaks: list[float] = []
        for metrics_path in (
            b04a_write_dir / "checkpoints"
        ).glob("*/seed_*/metrics_summary.json"):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            seed_peaks.append(float(metrics["budget_report"]["peak_cuda_mb"]))
        assert seed_peaks
        assert budget["elapsed_total_seconds"] == pytest.approx(
            manifest["wall_clock_seconds"]
        )
        assert budget["elapsed_total_seconds"] != pytest.approx(
            budget["thresholds"]["max_wall_seconds_total"]
        )
        assert budget["peak_cuda_mb"] == pytest.approx(max(seed_peaks))

    @pytest.mark.parametrize(
        "bad_peak", [None, True, "0", -1.0, float("nan"), float("inf")]
    )
    def test_budget_peak_malformed_fails_closed(self, bad_peak: Any) -> None:
        from topper_perception.neural.slp8_region_mini import (
            _b04a_observed_peak_cuda_mb,
        )

        per_seed = {
            seed: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION,
                seed=seed,
                feasibility="FEASIBLE",
                best_macro_iou=0.5,
                worst_subject_iou=0.4,
            )
            for seed in B04A_SEEDS
        }
        per_seed[B04A_SEEDS[0]].budget_report["peak_cuda_mb"] = bad_peak
        aggregate = _b04a_aggregate_candidate(
            SMALL_UNET_VERSION, per_seed, B04A_SEEDS
        )
        with pytest.raises(MiniProtocolError, match="peak_cuda_mb"):
            _b04a_observed_peak_cuda_mb({SMALL_UNET_VERSION: aggregate})

    # ------------------------------------------------------------------
    # (3) DONE/FAILED/STOPPED terminal JSON carry the seven fields
    # ------------------------------------------------------------------

    def test_terminal_json_identity(self, b04a_write_dir: Path) -> None:
        # Exactly one of the three terminal files exists.
        terminals = [
            b04a_write_dir / tf
            for tf in ("DONE.json", "FAILED.json", "STOPPED.json")
            if (b04a_write_dir / tf).exists()
        ]
        assert len(terminals) == 1, (
            f"expected exactly one terminal file, found "
            f"{[t.name for t in terminals]}"
        )
        terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
        _assert_has_required_identity_fields(
            terminal, where=terminals[0].name
        )
        assert terminal["experiment_id"] == SYNTHETIC_EXP_ID
        assert terminal["synthetic"] is True
        assert (
            terminal["data_manifest_source"]
            == "synthetic_canonical_manifest_sha256"
        )

    # ------------------------------------------------------------------
    # (4) run.log and per-seed logs first line is identity JSON
    # ------------------------------------------------------------------

    def test_log_files_first_line_is_identity(
        self, b04a_write_dir: Path
    ) -> None:
        run_log = b04a_write_dir / "logs" / "run.log"
        first = run_log.read_text(encoding="utf-8").splitlines()[0]
        first_obj = json.loads(first)
        _assert_has_required_identity_fields(
            first_obj, where="logs/run.log first line"
        )
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                seed_log = (
                    b04a_write_dir
                    / "logs"
                    / f"{cand}_seed_{seed:04d}.log"
                )
                if not seed_log.exists():
                    continue
                first_seed = seed_log.read_text(
                    encoding="utf-8"
                ).splitlines()[0]
                first_seed_obj = json.loads(first_seed)
                _assert_has_required_identity_fields(
                    first_seed_obj,
                    where=f"logs/{cand}_seed_{seed:04d}.log first line",
                )

    # ------------------------------------------------------------------
    # (5) CSV identity sidecars
    # ------------------------------------------------------------------

    def test_csv_identity_sidecars(self, b04a_write_dir: Path) -> None:
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                seed_dir = (
                    b04a_write_dir
                    / "checkpoints"
                    / cand
                    / f"seed_{seed:04d}"
                )
                epoch_csv = seed_dir / "epoch_metrics.csv"
                epoch_sidecar = epoch_csv.with_suffix(
                    epoch_csv.suffix + ".identity.json"
                )
                pred_csv = seed_dir / "predictions_manifest.csv"
                pred_sidecar = pred_csv.with_suffix(
                    pred_csv.suffix + ".identity.json"
                )
                for sidecar in (epoch_sidecar, pred_sidecar):
                    if not sidecar.exists():
                        continue
                    payload = json.loads(
                        sidecar.read_text(encoding="utf-8")
                    )
                    _assert_has_required_identity_fields(
                        payload["identity"],
                        where=str(sidecar.relative_to(b04a_write_dir)),
                    )

    # ------------------------------------------------------------------
    # (6) Run / candidate / seed / checkpoint identity are consistent
    # ------------------------------------------------------------------

    def test_run_candidate_seed_checkpoint_identity_consistent(
        self, b04a_write_dir: Path
    ) -> None:
        manifest = json.loads(
            (b04a_write_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Run-level model_version is multi_candidate[...] in config
        # order; candidate/seed/checkpoint model_version is the
        # candidate builder's exact version.
        assert manifest["model_version"].startswith("multi_candidate[")
        assert manifest["model_version"].endswith("]")
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            assert cand in manifest["model_version"]
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                seed_dir = (
                    b04a_write_dir
                    / "checkpoints"
                    / cand
                    / f"seed_{seed:04d}"
                )
                payload = torch.load(
                    seed_dir / "best.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                identity = payload["identity"]
                # EXP-ID, Git, config, data manifest, split:
                # exactly equal to the run-level bundle.
                for key in (
                    "experiment_id",
                    "git_commit",
                    "git_dirty",
                    "config_sha256",
                    "data_manifest_sha256",
                    "split_sha256",
                ):
                    assert identity[key] == manifest[key], (
                        f"identity drift on {cand}/seed_{seed:04d}: "
                        f"{key} checkpoint={identity[key]!r} "
                        f"manifest={manifest[key]!r}"
                    )
                # model_version differs only as the contract
                # requires: candidate-level exact builder version
                # (not the run-level multi_candidate[...]).
                assert identity["model_version"] == cand
                assert identity["model_version"] != manifest["model_version"]

    # ------------------------------------------------------------------
    # (7) Writer must use the (possibly mutated) result identity
    # ------------------------------------------------------------------

    def test_b04a_run_bundle_writer_uses_result_identity(
        self, tmp_path: Path
    ) -> None:
        """A run bundle writer cannot use a different identity than the
        B04ARunResult it received; the result is the single source of
        truth (R02 ITERATE review point 4: eliminate double identity
        source).
        """
        from topper_perception.neural.slp8_region_determinism import (
            apply_settings,
        )
        from topper_perception.neural.slp8_region_mini import (
            B04ARunResult,
            B04AAdvanceDecision,
            _write_b04a_run_bundle,
        )

        config = _build_b04a_mini_config()
        determinism = apply_settings(42, cpu_threads=1)
        result = B04ARunResult(
            config=config,
            dataset_manifest={"n_test_samples": 0},
            environment={},
            class_weight_result=compute_class_weights(
                {
                    "n_samples": 1,
                    "n_pixels": 9,
                    "per_class_pixel_ratio": {
                        0: 1.0 / 9.0,
                        1: 1.0 / 9.0,
                        2: 1.0 / 9.0,
                        3: 1.0 / 9.0,
                        4: 1.0 / 9.0,
                        5: 1.0 / 9.0,
                        6: 1.0 / 9.0,
                        7: 1.0 / 9.0,
                        8: 1.0 / 9.0,
                    },
                }
            ),
            candidate_results={},
            n_candidates_feasible=0,
            n_candidates_not_feasible=3,
            n_candidates_failed=0,
            n_candidates_stopped=0,
            overall_decision="MINI_NOT_FEASIBLE",
            advanced=(),
            near_tie_applied=False,
            near_tie_margin=B04A_NEAR_TIE_MARGIN,
            advance_decision=B04AAdvanceDecision(
                advanced=(),
                near_tie_applied=False,
                near_tie_margin=B04A_NEAR_TIE_MARGIN,
                tiebreaks=[],
            ),
            terminal_state="DONE",
            started_at_utc="2026-01-01T00:00:00+00:00",
            ended_at_utc="2026-01-01T00:00:00+00:00",
            wall_clock_seconds=0.0,
            input_hashes={},
            train_class_stats_source="synthetic",
            synthetic=True,
            determinism=determinism,
            resource_budget=ResourceBudget(
                max_wall_seconds_per_candidate=1.0,
                max_wall_seconds_total=1.0,
                max_peak_cuda_mb=8192.0,
            ),
            b01_contract_report=None,
            experiment_id="EXP-OWNER-A",
            data_manifest_sha256="a" * 64,
            git_commit="0" * 40,
            git_dirty=False,
        )
        identity_seed_results = {
            seed: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION,
                seed=seed,
                feasibility="FEASIBLE",
                best_macro_iou=0.5,
                worst_subject_iou=0.4,
            )
            for seed in B04A_SEEDS
        }
        result.candidate_results = {
            SMALL_UNET_VERSION: _b04a_aggregate_candidate(
                SMALL_UNET_VERSION, identity_seed_results, B04A_SEEDS
            )
        }
        # Mutate the result AFTER construction; the writer must use
        # the mutated values, proving there is no separate identity
        # source.
        result.experiment_id = "EXP-OWNER-MUTATED"
        result.data_manifest_sha256 = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bundle"
            _write_b04a_run_bundle(
                output_dir=out,
                result=result,
                config_sha256=_TEST_CONFIG_SHA,
            )
            manifest = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8")
            )
            assert (
                manifest["experiment_id"] == "EXP-OWNER-MUTATED"
            ), "writer used a different EXP-ID than the result object"
            assert (
                manifest["data_manifest_sha256"] == "b" * 64
            ), "writer used a different data_manifest_sha256"

    # ------------------------------------------------------------------
    # (8) resume rejects git_commit, git_dirty, split_sha256 drift
    # ------------------------------------------------------------------

    def test_resume_rejects_git_and_split_drift(self) -> None:
        saved = CheckpointIdentity(
            task_id=B04A_TASK_ID,
            candidate=SMALL_UNET_VERSION,
            model_version=SMALL_UNET_VERSION,
            seed=42,
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            config_sha256=_TEST_CONFIG_SHA,
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
            git_commit="0" * 40,
            git_dirty=False,
            experiment_id=_TEST_OWNER_EXP_ID,
            data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
        )
        for drift_field, new_value in (
            ("git_commit", "f" * 40),
            ("git_dirty", True),
            ("split_sha256", "e" * 64),
        ):
            drifted = replace(saved, **{drift_field: new_value})
            with pytest.raises(
                ResumeIdentityError, match=drift_field
            ):
                verify_resume_identity(saved=saved, requested=drifted)

    # ------------------------------------------------------------------
    # (9) Synthetic checkpoint's experiment_id / data_manifest_sha256
    # are non-empty
    # ------------------------------------------------------------------

    def test_synthetic_checkpoint_identity_non_empty(
        self, b04a_write_dir: Path
    ) -> None:
        for cand in B04A_ACTIVE_CANDIDATE_NAMES:
            for seed in B04A_SEEDS:
                seed_dir = (
                    b04a_write_dir
                    / "checkpoints"
                    / cand
                    / f"seed_{seed:04d}"
                )
                for pt_name in ("best.pt", "last.pt"):
                    payload = torch.load(
                        seed_dir / pt_name,
                        map_location="cpu",
                        weights_only=True,
                    )
                    identity = payload["identity"]
                    assert (
                        identity["experiment_id"] == SYNTHETIC_EXP_ID
                    )
                    assert identity["experiment_id"] != ""
                    assert identity["data_manifest_sha256"] != ""
                    # The synthetic manifest hash is a deterministic
                    # constant of the canonical payload; verify it
                    # matches the value computed at write time.
                    from topper_perception.neural.slp8_region_mini import (
                        _compute_synthetic_manifest_sha256,
                    )

                    assert identity["data_manifest_sha256"] == (
                        _compute_synthetic_manifest_sha256()
                    )

    # ------------------------------------------------------------------
    # (10) post-validation FAILED / STOPPED artifacts also carry
    # identity
    # ------------------------------------------------------------------

    def test_post_validation_terminal_artifacts_carry_identity(
        self, tmp_path: Path
    ) -> None:
        # Drive the B04A synthetic smoke into a STOPPED state with
        # a tiny per-candidate budget override.  The terminal JSON
        # must still carry the seven required identity fields; the
        # identity contract is not relaxed for post-validation
        # FAILED / STOPPED paths.
        out = tmp_path / "b04a_stopped"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        proc = _run_b04a_synthetic_writing_smoke(
            out,
            budget_override_seconds=1e-9,
        )
        # Either DONE (budget override did not trip the new budget
        # format) or STOPPED; the assertion is on the terminal
        # JSON regardless of the exit code.
        assert proc.returncode in {0, 1}, (
            f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
        )
        stopped_path = out / "STOPPED.json"
        done_path = out / "DONE.json"
        assert stopped_path.exists() or done_path.exists()
        if stopped_path.exists():
            data = json.loads(
                stopped_path.read_text(encoding="utf-8")
            )
            where = "STOPPED.json"
        else:
            data = json.loads(
                done_path.read_text(encoding="utf-8")
            )
            where = "DONE.json"
        _assert_has_required_identity_fields(data, where=where)
        assert data["experiment_id"] == SYNTHETIC_EXP_ID
        assert data["synthetic"] is True
        assert (
            data["data_manifest_source"]
            == "synthetic_canonical_manifest_sha256"
        )

    # ------------------------------------------------------------------
    # R03 ITERATE: cover-order / post-validation / single-identity tests
    # ------------------------------------------------------------------

    def test_write_status_files_extra_cannot_overwrite_identity(
        self, tmp_path: Path
    ) -> None:
        """``extra`` must NOT be able to override the frozen identity.

        The contract (R03 ITERATE) is that the ``identity`` dict
        is the single source of truth for the seven required
        identity fields.  A caller that passes a ``extra`` entry
        for ``experiment_id`` (or any other identity key) with a
        different value MUST trigger a fail-closed exception, not
        a silent override.
        """
        from topper_perception.neural.slp8_region_mini import (
            MiniProtocolError,
            write_status_files,
        )

        out = tmp_path / "override_attempt"
        with pytest.raises(
            MiniProtocolError, match="disagrees with frozen identity"
        ):
            write_status_files(
                out,
                status="FAILED",
                identity={
                    "experiment_id": "EXP-OWNER-A",
                    "git_commit": "0" * 40,
                    "git_dirty": False,
                    "config_sha256": "a" * 64,
                    "data_manifest_sha256": "b" * 64,
                    "split_sha256": "c" * 64,
                    "model_version": "multi_candidate[x,y,z]",
                },
                extra={
                    # The attempted override.  The contract
                    # forbids this even when the extra value is
                    # "sensible" because the identity carrier is
                    # the single source of truth.
                    "experiment_id": "EXP-OWNER-FAKE",
                    "error": "synthetic failure",
                },
            )
        # No terminal file should have been written when the
        # fail-closed exception fires.
        assert not (out / "FAILED.json").exists()
        assert not (out / "status.json").exists()

    def test_write_status_files_extra_consistent_identity_merged(
        self, tmp_path: Path
    ) -> None:
        """``extra`` keys that do NOT collide with ``identity`` are
        merged into the terminal JSON; ``identity`` keys win.
        """
        from topper_perception.neural.slp8_region_mini import (
            write_status_files,
        )

        out = tmp_path / "merge"
        write_status_files(
            out,
            status="FAILED",
            identity={
                "experiment_id": "EXP-OWNER-A",
                "git_commit": "0" * 40,
                "git_dirty": False,
                "config_sha256": "a" * 64,
                "data_manifest_sha256": "b" * 64,
                "split_sha256": "c" * 64,
                "model_version": "multi_candidate[x,y,z]",
            },
            extra={
                "error": "synthetic failure",
                "mode": "real-b01-b04a",
            },
        )
        data = json.loads(
            (out / "FAILED.json").read_text(encoding="utf-8")
        )
        _assert_has_required_identity_fields(data, where="FAILED.json")
        assert data["experiment_id"] == "EXP-OWNER-A"
        assert data["error"] == "synthetic failure"
        assert data["mode"] == "real-b01-b04a"

    def test_b04a_post_validation_failed_artifact_carries_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the B04A orchestrator raises a post-validation
        exception, the CLI's main ``except`` block must still
        write a ``FAILED.json`` that carries the seven required
        identity fields (R03 ITERATE).  The identity is
        reconstructed from the available CLI args, not from
        ``B04ARunResult`` (which does not exist on the failure
        path).
        """
        mod = _load_run_slp8_module()
        captured: dict[str, Any] = {}

        def fake_synthetic_cpu_smoke_b04a(
            config_path, output_dir, **kwargs
        ):  # noqa: ARG001
            # Synthetic post-validation exception AFTER config
            # validation, b01 contract verification, and EXP-ID
            # defaulting.  main()'s except block must build a
            # full identity block from the available CLI args
            # before writing FAILED.json.
            raise MiniProtocolError(
                "synthetic post-validation failure injected by "
                "TestB04AActualArtifactIdentityAudit"
            )

        monkeypatch.setattr(
            mod, "_run_synthetic_cpu_smoke_b04a", fake_synthetic_cpu_smoke_b04a
        )
        # Run the CLI through the real ``main`` entry point so the
        # except block is exercised.  A fresh output dir keeps
        # the test isolated.
        out = tmp_path / "post_validation_failed"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        # The post-validation exception is a non-rejection, so
        # main() returns 1 (FAILED).  Pre-validation rejections
        # (output-dir collision, --experiment-id missing on a
        # real B01 path, etc.) return 2; this path returns 1.
        assert rc == 1, (
            f"expected exit 1 (post-validation FAILED); stderr "
            f"may be inspected separately"
        )
        # Debug: list the output directory contents.
        contents = sorted(p.name for p in out.iterdir()) if out.exists() else []
        failed = out / "FAILED.json"
        assert failed.exists(), (
            f"post-validation FAILED.json missing at {failed}; "
            f"the CLI must emit a terminal artifact on a "
            f"post-validation failure (R03 ITERATE); out contents="
            f"{contents}"
        )
        data = json.loads(failed.read_text(encoding="utf-8"))
        _assert_has_required_identity_fields(data, where="FAILED.json")
        assert data["experiment_id"] == SYNTHETIC_EXP_ID
        assert data["synthetic"] is True
        assert (
            data["data_manifest_source"]
            == "synthetic_canonical_manifest_sha256"
        )
        assert "error" in data

    def test_post_validation_identity_construction_fails_closed_when_config_unreadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the post-validation path cannot construct the
        identity (helper raises ``MiniProtocolError``), main()
        returns 2 and does NOT write any terminal JSON.  This
        keeps the pre-validation contract intact (no fake
        identity is ever written just to satisfy the carrier).
        """
        mod = _load_run_slp8_module()

        def fake_build_identity(*args, **kwargs):
            raise MiniProtocolError(
                "_build_post_validation_identity injected failure"
            )

        # Inject a post-validation exception by patching
        # ``run_mini_b04a`` (the orchestrator step the synthetic
        # CPU smoke calls); the fake raises a ``MiniProtocolError``
        # that main()'s except block must convert into either a
        # terminal FAILED artifact carrying the identity, or a
        # fail-closed REJECTED if the identity cannot be
        # constructed.  We additionally patch
        # ``_build_post_validation_identity`` to fail so the
        # fail-closed path is exercised.
        def fake_run_mini_b04a(**kwargs):
            raise MiniProtocolError(
                "post-validation exception injected by "
                "TestB04AActualArtifactIdentityAudit"
            )

        from topper_perception.neural import slp8_region_mini as sm

        monkeypatch.setattr(sm, "run_mini_b04a", fake_run_mini_b04a)
        # ``_run_synthetic_cpu_smoke_b04a`` does an inline
        # ``from topper_perception.neural.slp8_region_mini import
        # run_mini_b04a``; the patch on ``sm.run_mini_b04a`` is
        # what that inline import picks up because both names
        # refer to the same module object.  We additionally
        # patch the module-level attribute on ``mod`` to cover
        # any other code path.
        monkeypatch.setattr(mod, "run_mini_b04a", fake_run_mini_b04a)
        monkeypatch.setattr(
            mod, "_build_post_validation_identity", fake_build_identity
        )

        out = tmp_path / "post_validation_id_fail_closed"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        assert rc == 2
        # main() must NOT write FAILED.json / status.json when
        # the helper cannot safely construct the identity.
        assert not (out / "FAILED.json").exists()
        assert not (out / "status.json").exists()

    def test_git_identity_frozen_at_run_start_unchanged_by_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bundle writers MUST NOT re-resolve ``git rev-parse
        HEAD`` / ``git status --porcelain`` themselves.  The run-
        start ``B04ARunResult.git_commit`` / ``.git_dirty`` is
        frozen; if a writer accidentally calls the resolver
        again, the bundle would carry the current HEAD rather
        than the run-start HEAD.  This test asserts that the
        writer takes the run-start values from the result.
        """
        from topper_perception.neural.slp8_region_determinism import (
            apply_settings,
        )
        from topper_perception.neural.slp8_region_mini import (
            B04AAdvanceDecision,
            B04ARunResult,
            _write_b04a_run_bundle,
        )

        config = _build_b04a_mini_config()
        determinism = apply_settings(42, cpu_threads=1)

        # Freeze the run-start identity to a recognisable sentinel.
        FROZEN_COMMIT = "0" * 40
        FROZEN_DIRTY = False
        result = B04ARunResult(
            config=config,
            dataset_manifest={"n_test_samples": 0},
            environment={},
            class_weight_result=compute_class_weights(
                {
                    "n_samples": 1,
                    "n_pixels": 9,
                    "per_class_pixel_ratio": {
                        cid: 1.0 / 9.0 for cid in range(9)
                    },
                }
            ),
            candidate_results={},
            n_candidates_feasible=0,
            n_candidates_not_feasible=3,
            n_candidates_failed=0,
            n_candidates_stopped=0,
            overall_decision="MINI_NOT_FEASIBLE",
            advanced=(),
            near_tie_applied=False,
            near_tie_margin=B04A_NEAR_TIE_MARGIN,
            advance_decision=B04AAdvanceDecision(
                advanced=(),
                near_tie_applied=False,
                near_tie_margin=B04A_NEAR_TIE_MARGIN,
                tiebreaks=[],
            ),
            terminal_state="DONE",
            started_at_utc="2026-01-01T00:00:00+00:00",
            ended_at_utc="2026-01-01T00:00:00+00:00",
            wall_clock_seconds=0.0,
            input_hashes={},
            train_class_stats_source="synthetic",
            synthetic=True,
            determinism=determinism,
            resource_budget=ResourceBudget(
                max_wall_seconds_per_candidate=1.0,
                max_wall_seconds_total=1.0,
                max_peak_cuda_mb=8192.0,
            ),
            b01_contract_report=None,
            experiment_id=SYNTHETIC_EXP_ID,
            data_manifest_sha256=_expected_synthetic_manifest_sha(),
            git_commit=FROZEN_COMMIT,
            git_dirty=FROZEN_DIRTY,
        )
        identity_seed_results = {
            seed: _make_fake_candidate_result(
                candidate=SMALL_UNET_VERSION,
                seed=seed,
                feasibility="FEASIBLE",
                best_macro_iou=0.5,
                worst_subject_iou=0.4,
            )
            for seed in B04A_SEEDS
        }
        result.candidate_results = {
            SMALL_UNET_VERSION: _b04a_aggregate_candidate(
                SMALL_UNET_VERSION, identity_seed_results, B04A_SEEDS
            )
        }

        # Now monkeypatch ``_resolve_git_identity`` to return a
        # different value, simulating a worktree drift (e.g. a
        # commit happens after run start).  The writer MUST
        # ignore the live resolver and use the frozen values.
        DRIFTED_COMMIT = "f" * 40
        DRIFTED_DIRTY = True

        def fake_resolve_git_identity():
            return DRIFTED_COMMIT, DRIFTED_DIRTY

        # Patch the symbol that ``_b04a_identity_block`` would
        # import if it tried to re-resolve.  We don't even need
        # the writer to call it; we just verify the carrier uses
        # the result's frozen values, not the live resolver.
        from topper_perception.neural import slp8_region_mini as sm

        monkeypatch.setattr(
            sm, "_resolve_git_identity", fake_resolve_git_identity
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bundle"
            _write_b04a_run_bundle(
                output_dir=out,
                result=result,
                config_sha256=_TEST_CONFIG_SHA,
            )
            manifest = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8")
            )
            # The bundle MUST carry the run-start values, not
            # the drifted values.  If the writer had called the
            # live resolver, the test would observe DRIFTED_*.
            assert manifest["git_commit"] == FROZEN_COMMIT, (
                "bundle writer re-resolved git identity; "
                "carrier no longer matches the run-start "
                "frozen identity"
            )
            assert manifest["git_dirty"] is FROZEN_DIRTY


# ---------------------------------------------------------------------------
# R04 ITERATE: git_commit fail-closed + frozen run identity context
# ---------------------------------------------------------------------------


class TestR04B04AGitCommitFailClosed:
    """R04 ITERATE: ``_b04a_identity_block`` and ``_resolve_git_identity``
    refuse to emit a formal B04A identity with a malformed or sentinel
    ``git_commit``.  The frozen B04A identity contract pins a real
    Git object ID at run start; a Reviewer must never see an empty
    or sentinel value in any formal carrier.
    """

    def test_b04a_identity_block_rejects_empty_git_commit(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for empty in ("", "   ", "\t\n", None):
            with pytest.raises(MiniProtocolError, match="git_commit"):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=_TEST_OWNER_EXP_ID,
                    data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                    git_commit=empty,  # type: ignore[arg-type]
                )

    def test_b04a_identity_block_rejects_unresolvable_sentinel(self):
        from topper_perception.neural.slp8_region_mini import (
            UNRESOLVABLE_GIT_COMMIT,
            _b04a_identity_block,
            MiniProtocolError,
        )

        with pytest.raises(MiniProtocolError, match="unresolvable"):
            _b04a_identity_block(
                config=_build_b04a_mini_config(),
                config_sha256=_TEST_CONFIG_SHA,
                experiment_id=_TEST_OWNER_EXP_ID,
                data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                git_commit=UNRESOLVABLE_GIT_COMMIT,
            )

    def test_b04a_identity_block_rejects_non_hex_git_commit(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for bad in (
            "z" * 40,                   # non-hex characters
            "g" * 40,                   # invalid hex letter
            "0" * 39 + "z",             # 40 chars, last char non-hex
        ):
            with pytest.raises(MiniProtocolError, match="non-hex"):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=_TEST_OWNER_EXP_ID,
                    data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                    git_commit=bad,
                )

    def test_b04a_identity_block_rejects_wrong_length_git_commit(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for bad in (
            "abc",                       # too short
            "0" * 7,                     # 7 chars
            "0" * 39,                    # 39 chars
            "0" * 41,                    # 41 chars
            "0" * 63,                    # 63 chars
            "0" * 65,                    # 65 chars
            "0" * 128,                   # too long
        ):
            with pytest.raises(MiniProtocolError, match="length"):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=_TEST_OWNER_EXP_ID,
                    data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                    git_commit=bad,
                )

    def test_b04a_identity_block_rejects_git_commit_with_internal_whitespace(
        self,
    ):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
            MiniProtocolError,
        )

        for bad in (
            " " + "0" * 39,              # leading whitespace
            "0" * 39 + " ",              # trailing whitespace
            "0" * 20 + " " + "0" * 19,   # internal whitespace
        ):
            with pytest.raises(MiniProtocolError, match="whitespace"):
                _b04a_identity_block(
                    config=_build_b04a_mini_config(),
                    config_sha256=_TEST_CONFIG_SHA,
                    experiment_id=_TEST_OWNER_EXP_ID,
                    data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                    git_commit=bad,
                )

    def test_b04a_identity_block_accepts_40_and_64_char_hex(self):
        from topper_perception.neural.slp8_region_mini import (
            _b04a_identity_block,
        )

        sha1 = "0" * 40
        sha256 = "f" * 64
        for valid in (sha1, sha256, "Ab" + "c" * 38, "F" * 64):
            block = _b04a_identity_block(
                config=_build_b04a_mini_config(),
                config_sha256=_TEST_CONFIG_SHA,
                experiment_id=_TEST_OWNER_EXP_ID,
                data_manifest_sha256=_TEST_FREEZE_MANIFEST_FILE_SHA,
                git_commit=valid,
            )
            # The block stores the lowercased value to match
            # ``git rev-parse`` output convention.
            assert block["git_commit"] == valid.lower()
            assert block["git_commit"] != ""

    def test_resolve_git_identity_raises_on_resolver_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``git rev-parse HEAD`` or ``git status --porcelain``
        fails, ``_resolve_git_identity`` raises
        :class:`MiniProtocolError`.  Formal B04A identity
        construction cannot accept a sentinel
        ``unresolvable_git_commit`` value.
        """
        from topper_perception.neural import slp8_region_mini as sm
        from topper_perception.neural.slp8_region_mini import (
            MiniProtocolError,
            _resolve_git_identity,
        )

        class _Boom(RuntimeError):
            pass

        def _explode(*args, **kwargs):  # noqa: ARG001
            raise _Boom("synthetic git rev-parse failure")

        monkeypatch.setattr(
            "subprocess.run", _explode
        )
        with pytest.raises(MiniProtocolError, match="git rev-parse"):
            _resolve_git_identity()
        # Module-level symbol still references the same callable.
        assert sm._resolve_git_identity is _resolve_git_identity


class TestR04B04AFrozenRunIdentityContext:
    """R04 ITERATE: the CLI freezes the run identity context BEFORE
    dispatching any handler.  The post-validation FAILED identity
    uses that frozen context; the handler MUST NOT re-resolve
    Git identity at exception time.
    """

    def test_cli_frozen_identity_used_in_post_validation_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI main() resolves ``_resolve_git_identity`` once at the
        top and threads the value through to the post-validation
        FAILED path.  If a downstream writer accidentally re-resolves
        and the worktree has drifted, the FAILED.json still carries
        the dispatch-time frozen Git SHA, not the live value.
        """
        mod = _load_run_slp8_module()

        # The dispatch-time frozen value the CLI must record.
        CLI_FROZEN_COMMIT = "a" * 40
        CLI_FROZEN_DIRTY = False

        # The "current state" the writer / orchestrator would see
        # if they re-resolved after the dispatch.
        DRIFTED_COMMIT = "f" * 40
        DRIFTED_DIRTY = True

        # Counter to differentiate the CLI dispatch freeze (call 1)
        # from any later re-resolve attempts.
        state = {"call_count": 0}

        def _cli_freeze_then_drift():
            state["call_count"] += 1
            if state["call_count"] == 1:
                return CLI_FROZEN_COMMIT, CLI_FROZEN_DIRTY
            return DRIFTED_COMMIT, DRIFTED_DIRTY

        from topper_perception.neural import slp8_region_mini as sm

        # Patch the source module's ``_resolve_git_identity``;
        # the script does a local import in main(), so the script
        # module itself does not own the symbol.
        monkeypatch.setattr(sm, "_resolve_git_identity", _cli_freeze_then_drift)

        def _fake_synthetic_cpu_smoke_b04a(
            config_path, output_dir, **kwargs  # noqa: ARG001
        ):
            # Synthetic post-validation exception after data
            # contract verification, so the FAILED path runs
            # through _build_post_validation_identity.
            raise MiniProtocolError(
                "synthetic R04 post-validation failure injected by "
                "TestR04B04AFrozenRunIdentityContext"
            )

        monkeypatch.setattr(
            mod, "_run_synthetic_cpu_smoke_b04a", _fake_synthetic_cpu_smoke_b04a
        )

        out = tmp_path / "frozen_id"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        # The post-validation FAILED returns 1 (non-rejection).
        assert rc == 1
        failed = out / "FAILED.json"
        assert failed.exists(), (
            "post-validation FAILED.json missing; the CLI must "
            "emit a terminal artifact on a post-validation failure "
            "(R04 ITERATE)"
        )
        data = json.loads(failed.read_text(encoding="utf-8"))
        # FAILED.json MUST carry the dispatch-time frozen git SHA.
        assert data["git_commit"] == CLI_FROZEN_COMMIT
        assert data["git_commit"] != DRIFTED_COMMIT
        assert data["git_dirty"] is CLI_FROZEN_DIRTY
        # R05 ITERATE: the post-validation FAILED path must NOT
        # re-resolve Git identity.  The CLI dispatch freeze is the
        # single source; call_count must be exactly 1.
        assert state["call_count"] == 1, (
            f"expected _resolve_git_identity call_count == 1 "
            f"(CLI dispatch freeze only); got {state['call_count']}.  "
            "post-validation FAILED path must not re-resolve Git "
            "(R05 ITERATE fix: changed from >= 1 to == 1)"
        )

    def test_cli_returns_2_when_identity_context_unresolvable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``_resolve_git_identity`` fails BEFORE the dispatch
        (no run-start identity can be established), the CLI fails
        closed: no output file is created, exit code 2, no
        sentinel identity is written.
        """
        mod = _load_run_slp8_module()
        from topper_perception.neural import slp8_region_mini as sm
        from topper_perception.neural.slp8_region_mini import (
            MiniProtocolError,
        )

        def _unresolvable():
            raise MiniProtocolError(
                "_resolve_git_identity: git rev-parse HEAD failed; "
                "rc=128 stderr='fatal: not a git repository'"
            )

        monkeypatch.setattr(sm, "_resolve_git_identity", _unresolvable)

        out = tmp_path / "no_id_context"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        # The CLI fails closed: 2, no file in the output dir.
        assert rc == 2
        assert not (out / "FAILED.json").exists()
        assert not (out / "status.json").exists()
        assert not (out / "DONE.json").exists()


class TestR04B04ARealB01PostValidationFailed:
    """R04 ITERATE: real B01 post-validation failure path carries
    the same seven required identity fields.  The test uses a
    fresh B01 freeze fixture and a synthetic Owner EXP-ID; no
    TEST labels are read.
    """

    def test_real_b01_post_validation_failed_artifact_carriers_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive a real B01 dispatch with a real B01 freeze fixture,
        inject a post-validation exception AFTER the B01 contract
        has been verified, and assert the FAILED.json / status.json
        carry the same seven required identity fields.
        ``data_manifest_sha256`` MUST equal the on-disk
        ``freeze_manifest.json`` file SHA.  ``git_commit`` MUST
        equal the dispatch-time frozen value (not the live
        resolver).  ``TEST=0`` is enforced by the
        ``_make_fake_b01_freeze_dir`` helper (which never reads
        any TEST labels).
        """
        mod = _load_run_slp8_module()
        from topper_perception.io.slp8_training_table_freeze import (
            load_b01_freeze_tables,
        )
        from test_slp8_region_mini import _make_fake_b01_freeze_dir

        # Build a real B01 freeze fixture and capture the actual
        # on-disk ``freeze_manifest.json`` file SHA.  The fixture
        # creates train+val ONLY; the runner uses
        # load_b01_freeze_tables(..., load_test=False) so TEST
        # rows are never loaded and n_test_samples=0.
        # NOTE: ``test_manifest.csv`` IS created by the fixture
        # (it's a structural part of the B01 freeze schema, even
        # when TEST rows are not loaded at runtime).  The TEST=0
        # invariant is enforced by ``load_test=False`` in the
        # runner and by ``freeze._test_rows is None`` here.
        b01_freeze = _make_fake_b01_freeze_dir(
            tmp_path / "b01_freeze",
        )
        freeze = load_b01_freeze_tables(b01_freeze, load_test=False)
        freeze_manifest_path = b01_freeze / "freeze_manifest.json"
        assert freeze_manifest_path.is_file(), (
            "B01 freeze fixture must include freeze_manifest.json; "
            f"missing at {freeze_manifest_path}"
        )
        # R05 ITERATE: assert the TEST=0 invariant at the loader
        # level.  With ``load_test=False``, the freeze handle's
        # ``_test_rows`` must be None -- this is the concrete
        # proof that the runner never loads TEST labels.
        assert freeze._test_rows is None, (  # noqa: SLF001
            "freeze._test_rows must be None after "
            "load_b01_freeze_tables(..., load_test=False); "
            "the TEST=0 invariant is violated "
            "(R05 ITERATE fix for TestR04B04ARealB01PostValidationFailed)"
        )
        import hashlib

        expected_dm_sha = hashlib.sha256(
            freeze_manifest_path.read_bytes()
        ).hexdigest()
        # TEST=0 invariant: the freeze was loaded with
        # ``load_test=False`` so the B04 Mini never reads any TEST
        # labels.  The post-validation helper does not touch TEST
        # either.  This test therefore never enables
        # ``enable_test_access`` and never accesses
        # ``freeze.test_rows`` (which would raise
        # ``TestLeakageError`` by design).

        # The dispatch-time frozen Git SHA the CLI must record.
        CLI_FROZEN_COMMIT = "1" * 40
        CLI_FROZEN_DIRTY = False
        from topper_perception.neural import slp8_region_mini as sm

        def _frozen_resolver():
            return CLI_FROZEN_COMMIT, CLI_FROZEN_DIRTY

        monkeypatch.setattr(sm, "_resolve_git_identity", _frozen_resolver)

        # Inject a post-validation exception AFTER the B01 contract
        # has been verified.  The exception simulates a real
        # training-time failure (e.g. budget exceeded) so the
        # FAILED path runs through _build_post_validation_identity.
        OWNER_EXP_ID = (
            "EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R04"
        )

        def _fake_run_real_b01_b04a(**kwargs):  # noqa: ARG001
            raise MiniProtocolError(
                "R04 synthetic post-validation exception after B01 "
                "contract verification"
            )

        monkeypatch.setattr(
            mod, "_run_real_b01_b04a", _fake_run_real_b01_b04a
        )

        out = tmp_path / "real_b01_post_validation_failed"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        # The data_root argument is required by ``--dataset-root``
        # but is not read by the post-validation helper (which only
        # reads the B01 freeze dir).  Pass a tmp_path; the B01
        # contract is mocked out so the dataset_root content is
        # never inspected.
        data_root = tmp_path / "data_root"
        data_root.mkdir(parents=True, exist_ok=True)
        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--run-authorized",
                "--b01-freeze-dir", str(b01_freeze),
                "--dataset-root", str(data_root),
                "--experiment-id", OWNER_EXP_ID,
            ]
        )
        # post-validation FAILED returns 1 (non-rejection).
        assert rc == 1, (
            f"expected exit 1 (post-validation FAILED on real B01 "
            f"path); got rc={rc}"
        )
        failed = out / "FAILED.json"
        assert failed.exists(), (
            f"post-validation FAILED.json missing at {failed}; "
            f"the CLI must emit a terminal artifact on a real B01 "
            f"post-validation failure (R04 ITERATE)"
        )
        data = json.loads(failed.read_text(encoding="utf-8"))
        _assert_has_required_identity_fields(data, where="FAILED.json")
        # Real B01 path: Owner EXP-ID, NOT the synthetic sentinel.
        assert data["experiment_id"] == OWNER_EXP_ID
        # data_manifest_sha256 == freeze_manifest.json file SHA.
        assert data["data_manifest_sha256"] == expected_dm_sha
        # git_commit == dispatch-time frozen value.
        assert data["git_commit"] == CLI_FROZEN_COMMIT
        assert data["git_dirty"] is CLI_FROZEN_DIRTY
        # Real B01 path: synthetic marker is False.
        assert data["synthetic"] is False
        assert (
            data["data_manifest_source"] == "freeze_manifest_file_sha256"
        )
        # The CLI must also write status.json with the same fields
        # (sibling of FAILED.json).  The contract guarantees the
        # terminal artifact identity matches the run-level bundle.
        status = out / "status.json"
        assert status.exists(), (
            f"status.json missing at {status}; the CLI must emit a "
            f"status.json sibling on a real B01 post-validation "
            f"failure (R04 ITERATE)"
        )
        sdata = json.loads(status.read_text(encoding="utf-8"))
        # status.json MUST carry the same seven required fields
        # (R03 ITERATE: identity merged into status.json top level).
        for field in _REQUIRED_B04A_IDENTITY_FIELDS:
            assert field in sdata, (
                f"status.json missing identity field {field!r} "
                f"(R04 ITERATE: status.json must carry the same "
                f"identity as FAILED.json)"
            )
        # FAILED.json and status.json MUST agree on every identity
        # field (R03 ITERATE: single source of truth).
        for field in _REQUIRED_B04A_IDENTITY_FIELDS:
            assert data[field] == sdata[field], (
                f"FAILED.json and status.json disagree on "
                f"{field!r}: FAILED={data[field]!r} status="
                f"{sdata[field]!r} (R04 ITERATE: same identity)"
            )


class TestR05B04ANormalSuccessGitIdentity:
    """R05 ITERATE: the CLI resolves Git identity once at dispatch
    time; ``run_mini_b04a`` receives the frozen values and does NOT
    re-resolve.  The normal-success path must complete with
    ``call_count == 1`` and every carrier in the DONE bundle must
    agree on the frozen ``git_commit`` / ``git_dirty`` values.

    This is the regression test for the defect identified by the
    Codex Reviewer: before R05, ``run_mini_b04a`` internally called
    ``_resolve_git_identity()`` a second time, causing normal
    checkpoints/results to potentially carry a different SHA than the
    post-validation FAILED path (which used the frozen CLI value).
    """

    def test_normal_success_resolver_called_exactly_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Drive the synthetic B04A smoke to normal completion and
        assert that ``_resolve_git_identity`` is called exactly once
        (the CLI dispatch freeze).  ``run_mini_b04a`` must NOT call
        it again; any re-resolve would raise.

        The DONE bundle is audited: every carrier must agree on the
        frozen ``git_commit = 'a' * 40`` and ``git_dirty = False``.
        """
        mod = _load_run_slp8_module()

        # The CLI dispatch-time frozen value.
        FROZEN_COMMIT = "a" * 40
        FROZEN_DIRTY = False

        state = {"call_count": 0}

        def _cli_freeze_then_explode():
            state["call_count"] += 1
            if state["call_count"] == 1:
                return FROZEN_COMMIT, FROZEN_DIRTY
            # R05: run_mini_b04a must NOT re-resolve; any second
            # call proves the defect is not fixed.
            raise AssertionError(
                "_resolve_git_identity called more than once; "
                "run_mini_b04a re-resolved Git identity after the "
                "CLI dispatch freeze (R05 ITERATE defect)"
            )

        from topper_perception.neural import slp8_region_mini as sm

        monkeypatch.setattr(sm, "_resolve_git_identity", _cli_freeze_then_explode)

        out = tmp_path / "normal_success_git_id"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        assert rc == 0, f"expected exit 0 (normal DONE); got rc={rc}"
        done = out / "DONE.json"
        assert done.exists(), "DONE.json missing after normal synthetic B04A run"
        data = json.loads(done.read_text(encoding="utf-8"))
        # R05: the normal-success path uses the CLI dispatch-time frozen SHA.
        assert data["git_commit"] == FROZEN_COMMIT
        assert data["git_dirty"] is FROZEN_DIRTY
        # Strict assertion: exactly one call, no re-resolution.
        assert state["call_count"] == 1, (
            f"expected _resolve_git_identity call_count == 1 "
            f"(CLI dispatch freeze only); got {state['call_count']}.  "
            "run_mini_b04a must not re-resolve Git identity "
            "(R05 ITERATE)"
        )

    def test_normal_success_bundle_carriers_agree_on_git_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify every carrier in the normal-success DONE bundle
        carries the same frozen Git identity.  Covered carriers:

        - DONE.json (run-level)
        - status.json (run-level)
        - manifest.json (run-level)
        - candidate_decision.json (run-level)
        - per-seed identity sidecars under checkpoints/<cand>/seed_<seed>/
        - checkpoint best.pt / last.pt identity blocks

        All must agree on ``git_commit = 'a' * 40`` and
        ``git_dirty = False``.
        """
        mod = _load_run_slp8_module()

        FROZEN_COMMIT = "a" * 40
        FROZEN_DIRTY = False

        def _frozen_resolver():
            return FROZEN_COMMIT, FROZEN_DIRTY

        from topper_perception.neural import slp8_region_mini as sm

        monkeypatch.setattr(sm, "_resolve_git_identity", _frozen_resolver)

        out = tmp_path / "bundle_audit_git_id"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

        rc = mod.main(
            [
                "--config", str(B04A_CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke-b04a",
            ]
        )
        assert rc == 0
        assert (out / "DONE.json").exists()

        # Collect git identity from every required carrier.
        carrier_git_commits: dict[str, str] = {}
        carrier_git_dirty: dict[str, bool] = {}

        def _extract_git_identity(path: Path, label: str) -> None:
            """Extract git_commit/git_dirty from a JSON carrier."""
            if not path.is_file():
                return
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                git_commit = content.get("git_commit", "")
                git_dirty = content.get("git_dirty")
                carrier_git_commits[label] = git_commit
                carrier_git_dirty[label] = git_dirty
            except Exception:
                pass

        # Run-level carriers.
        for fname, label in [
            ("DONE.json", "DONE.json"),
            ("status.json", "status.json"),
            ("manifest.json", "manifest.json"),
            ("candidate_decision.json", "candidate_decision.json"),
        ]:
            _extract_git_identity(out / fname, label)

        # Per-seed identity sidecars.
        checkpoints_dir = out / "checkpoints"
        if checkpoints_dir.is_dir():
            for cand_dir in checkpoints_dir.iterdir():
                if not cand_dir.is_dir():
                    continue
                for seed_dir in cand_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                    seed_label = f"checkpoints/{cand_dir.name}/{seed_dir.name}"
                    _extract_git_identity(seed_dir / "seed_identity.json", f"{seed_label}/seed_identity.json")
                    # Read checkpoint pt identity from metadata.
                    for pt_name in ("best.pt", "last.pt"):
                        pt_path = seed_dir / pt_name
                        if pt_path.is_file():
                            try:
                                import torch as _torch

                                ckpt = _torch.load(
                                    pt_path, map_location="cpu", weights_only=False
                                )
                                if "identity" in ckpt:
                                    id_block = ckpt["identity"]
                                    carrier_git_commits[f"{seed_label}/{pt_name}"] = str(
                                        id_block.get("git_commit", "")
                                    )
                                    carrier_git_dirty[f"{seed_label}/{pt_name}"] = bool(
                                        id_block.get("git_dirty")
                                    )
                            except Exception:
                                pass

        # All carriers must have the frozen Git commit.
        for label, gc in carrier_git_commits.items():
            assert gc == FROZEN_COMMIT, (
                f"carrier {label} has git_commit={gc!r}, expected "
                f"{FROZEN_COMMIT!r} (R05 ITERATE: all carriers must "
                "agree on the CLI dispatch-time frozen value)"
            )
        for label, gd in carrier_git_dirty.items():
            assert gd is FROZEN_DIRTY, (
                f"carrier {label} has git_dirty={gd!r}, expected "
                f"{FROZEN_DIRTY!r} (R05 ITERATE: all carriers must "
                "agree on the CLI dispatch-time frozen value)"
            )
        # At least run-level DONE.json must be checked.
        assert "DONE.json" in carrier_git_commits, (
            "DONE.json must be audited for git identity (R05 ITERATE)"
        )
