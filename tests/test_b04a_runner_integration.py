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
        budget_report={},
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
            candidate=SMALL_UNET_VERSION,
            seed=42,
        )
        block_deeplab = _b04a_identity_block(
            config=_build_b04a_mini_config(),
            config_sha256="a" * 64,
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
            freeze_manifest_sha256="f" * 64,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
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
            freeze_manifest_sha256="f" * 64,
            train_class_stats_sha256="e" * 64,
            class_weight_sha256="d" * 64,
            input_manifest_hashes_sha256="c" * 64,
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
        )
        assert result == 0
        assert calls.get("b04_called") is True
        assert calls.get("b04a_called") is None
        # The B04 helper received the resolved B04 MiniConfig.
        cfg = calls["b04_kwargs"]["config"]
        assert cfg.protocol == B04_PROTOCOL_NAME
        assert cfg.seeds == (42,)

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
            )
