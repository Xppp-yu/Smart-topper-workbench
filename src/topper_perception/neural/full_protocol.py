"""Frozen P5.2-C Full fair-comparison protocol (no runner).

This module is the single source of truth for the *frozen* P5.2-C Full protocol:
data boundary, candidate set, outer subject-grouped CV, inner epoch-selection,
primary metric, and the fixed final-selection rule. It is deliberately **pure
stdlib** (no ``torch``, no NumPy, no PoPu I/O) so it can be imported, validated
and unit-tested on any machine without the optional ``neural`` dependency.

It does **not** run anything: there is no Full runner here. The corresponding
config is ``configs/experiments/popu_neural_full_v0.1.json`` and the frozen
prose is ``docs/stage_reports/P5_2_C_POPU_NEURAL_FULL_PROTOCOL_v0.1.md``.

P5.2-C Full 尚未运行 (the Full comparison has not been run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Data boundary (fail-closed)
# ---------------------------------------------------------------------------

DATASET = "popu_tactilus"
QUALITY_MANIFEST_PATH = "outputs/metrics/popu_tactilus_quality_results_v0.1.csv"
#: SHA-256 of the frozen P2 quality manifest (the ACCEPT-only cohort source).
QUALITY_MANIFEST_SHA256 = (
    "9d3398a587b183f7e27ea68ada2eda1e5e82ebadb2ac9caf7a74b5763d3e954c"
)
COHORT = "primary"

#: Expected full-cohort boundary. ``n_snapshots == n_records * snapshots_per_record``.
DATA_BOUNDARY: dict[str, int] = {
    "n_subjects": 60,
    "n_records": 5006,
    "n_snapshots": 50060,
    "snapshots_per_record": 10,
}

# ---------------------------------------------------------------------------
# Candidates (frozen)
# ---------------------------------------------------------------------------

NEURAL_CANDIDATES: tuple[str, ...] = ("matrix_mlp", "tiny_cnn", "small_resnet")
FROZEN_SVM_REFERENCE = "calibrated_linear_svm"

# ---------------------------------------------------------------------------
# Outer fair-evaluation protocol (reuse P5.1)
# ---------------------------------------------------------------------------

GROUP_KEY = "subject_id"
N_SPLITS = 5
N_REPEATS = 3
#: One subject-grouped fold set per seed -> exactly N_REPEATS repeats.
OUTER_SEEDS: tuple[int, ...] = (11, 22, 33)

# ---------------------------------------------------------------------------
# Inner epoch selection (Stage A), no leakage
# ---------------------------------------------------------------------------

INNER_N_SPLITS = 4
MAX_EPOCHS = 15
MIN_EPOCHS = 5
PATIENCE = 3
MONITOR = "val_loss"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# Resources and stop conditions
# ---------------------------------------------------------------------------

DEVICE = "cuda"
AMP_ENABLED = True
MAX_CUDA_MB = 8000
MAX_TOTAL_TRAIN_SECONDS = 21600

# ---------------------------------------------------------------------------
# Final selection rule (fixed)
# ---------------------------------------------------------------------------

SELECTION_CRITERION = "record_macro_f1_mean"
SELECTION_MARGIN = 0.005
SUBSTANTIAL_WORST_SUBJECT_F1 = 0.02
SUBSTANTIAL_WEAKEST_CLASS_F1 = 0.01
SUBSTANTIAL_STD_REDUCTION = 0.001


@dataclass(frozen=True, slots=True)
class FullCandidateResult:
    """One candidate's frozen summary row fed into :func:`select_full_winner`."""

    model: str
    passed_gate: bool
    is_frozen_svm: bool
    record_macro_f1_mean: float
    record_balanced_acc_mean: float
    worst_subject_macro_f1_mean: float
    record_macro_f1_std: float
    weakest_class_record_f1: float


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    """Fail closed when ``actual`` differs from the frozen ``expected``."""
    if actual != expected:
        raise ValueError(
            f"Full protocol violation: {label} is {actual!r}, expected {expected!r}."
        )


def validate_full_data_boundary(
    *,
    n_subjects: int,
    n_records: int,
    n_snapshots: int,
    snapshots_per_record: int,
) -> None:
    """Fail closed when the loaded full cohort diverges from the frozen boundary.

    Any mismatch — subject/record/snapshot count or snapshots-per-record — raises
    :class:`ValueError` so a governed run cannot silently train on a different
    cohort than the one the protocol froze.
    """
    _require_equal(int(n_subjects), DATA_BOUNDARY["n_subjects"], "n_subjects")
    _require_equal(int(n_records), DATA_BOUNDARY["n_records"], "n_records")
    _require_equal(
        int(snapshots_per_record),
        DATA_BOUNDARY["snapshots_per_record"],
        "snapshots_per_record",
    )
    _require_equal(int(n_snapshots), DATA_BOUNDARY["n_snapshots"], "n_snapshots")
    # Cross-consistency guard: the boundary is self-consistent.
    if int(n_snapshots) != int(n_records) * int(snapshots_per_record):
        raise ValueError(
            "Full protocol violation: n_snapshots "
            f"({n_snapshots}) != n_records ({n_records}) * snapshots_per_record "
            f"({snapshots_per_record})."
        )


def validate_full_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen Full config's structural values against this module.

    Complements :func:`topper_perception.experiments.contracts.validate_experiment_config`
    (schema-level) with the P5.2-C-specific frozen values. Any drift raises
    :class:`ValueError`; nothing is returned on success.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Full config must be a mapping.")

    _require_equal(config.get("scope"), "full", "scope")
    _require_equal(config.get("runner_type"), "popu_neural_full", "runner_type")

    manifests = config.get("data_manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise ValueError("Full config must pin exactly one data_manifests entry.")
    pinned_sha = str(manifests[0].get("sha256", "")).lower()
    _require_equal(pinned_sha, QUALITY_MANIFEST_SHA256, "data_manifests[0].sha256")

    params = config.get("parameters")
    if not isinstance(params, Mapping):
        raise ValueError("Full config parameters must be a mapping.")

    _require_equal(params.get("cohort"), COHORT, "parameters.cohort")
    _require_equal(
        str(params.get("quality_manifest_sha256", "")).lower(),
        QUALITY_MANIFEST_SHA256,
        "parameters.quality_manifest_sha256",
    )

    boundary = params.get("data_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Full config parameters.data_boundary is required.")
    validate_full_data_boundary(
        n_subjects=boundary["n_subjects"],
        n_records=boundary["n_records"],
        n_snapshots=boundary["n_snapshots"],
        snapshots_per_record=boundary["snapshots_per_record"],
    )

    candidates = params.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("Full config parameters.candidates is required.")
    _require_equal(tuple(candidates.get("neural", [])), NEURAL_CANDIDATES, "candidates.neural")
    _require_equal(
        candidates.get("frozen_svm_reference"),
        FROZEN_SVM_REFERENCE,
        "candidates.frozen_svm_reference",
    )

    # The SVM is a frozen reference: it must NOT appear among the (trained) model_configs.
    model_configs = params.get("model_configs", [])
    model_names = [entry["name"] for entry in model_configs]
    _require_equal(model_names, list(NEURAL_CANDIDATES), "parameters.model_configs names")

    outer = params.get("evaluation_protocol")
    if not isinstance(outer, Mapping):
        raise ValueError("Full config parameters.evaluation_protocol is required.")
    _require_equal(outer.get("group"), GROUP_KEY, "evaluation_protocol.group")
    _require_equal(int(outer.get("n_splits")), N_SPLITS, "evaluation_protocol.n_splits")
    _require_equal(int(outer.get("n_repeats")), N_REPEATS, "evaluation_protocol.n_repeats")
    _require_equal(list(outer.get("seeds", [])), list(OUTER_SEEDS), "evaluation_protocol.seeds")

    inner = params.get("inner_epoch_selection")
    if not isinstance(inner, Mapping):
        raise ValueError("Full config parameters.inner_epoch_selection is required.")
    _require_equal(int(inner.get("inner_n_splits")), INNER_N_SPLITS, "inner_epoch_selection.inner_n_splits")
    _require_equal(int(inner.get("max_epochs")), MAX_EPOCHS, "inner_epoch_selection.max_epochs")
    _require_equal(int(inner.get("min_epochs")), MIN_EPOCHS, "inner_epoch_selection.min_epochs")
    _require_equal(int(inner.get("patience")), PATIENCE, "inner_epoch_selection.patience")
    _require_equal(inner.get("monitor"), MONITOR, "inner_epoch_selection.monitor")
    optimizer = inner.get("optimizer")
    _require_equal(float(optimizer.get("lr")), LEARNING_RATE, "inner_epoch_selection.optimizer.lr")
    _require_equal(
        float(optimizer.get("weight_decay")),
        WEIGHT_DECAY,
        "inner_epoch_selection.optimizer.weight_decay",
    )

    resources = params.get("resources_and_stop")
    if not isinstance(resources, Mapping):
        raise ValueError("Full config parameters.resources_and_stop is required.")
    _require_equal(resources.get("device"), DEVICE, "resources_and_stop.device")
    _require_equal(bool(resources.get("amp_enabled")), AMP_ENABLED, "resources_and_stop.amp_enabled")
    _require_equal(int(resources.get("max_cuda_mb")), MAX_CUDA_MB, "resources_and_stop.max_cuda_mb")
    _require_equal(
        int(resources.get("max_total_train_seconds")),
        MAX_TOTAL_TRAIN_SECONDS,
        "resources_and_stop.max_total_train_seconds",
    )

    selection = params.get("model_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Full config parameters.model_selection is required.")
    _require_equal(selection.get("criterion"), SELECTION_CRITERION, "model_selection.criterion")
    _require_equal(float(selection.get("margin")), SELECTION_MARGIN, "model_selection.margin")
    improvement = selection.get("neural_vs_svm", {}).get("substantial_improvement", {})
    _require_equal(
        float(improvement.get("worst_subject_macro_f1_absolute_gain")),
        SUBSTANTIAL_WORST_SUBJECT_F1,
        "model_selection.substantial_improvement.worst_subject_macro_f1_absolute_gain",
    )
    _require_equal(
        float(improvement.get("weakest_class_record_f1_absolute_gain")),
        SUBSTANTIAL_WEAKEST_CLASS_F1,
        "model_selection.substantial_improvement.weakest_class_record_f1_absolute_gain",
    )
    _require_equal(
        float(improvement.get("record_macro_f1_std_absolute_reduction")),
        SUBSTANTIAL_STD_REDUCTION,
        "model_selection.substantial_improvement.record_macro_f1_std_absolute_reduction",
    )


def _neural_beats_svm(
    nn: FullCandidateResult,
    svm: FullCandidateResult,
    *,
    margin: float,
    substantial_worst_subject_f1: float,
    substantial_weakest_class_f1: float,
    substantial_std_reduction: float,
) -> bool:
    """Apply the pre-registered NN-vs-SVM rule (section 七 rule 5).

    ``True`` when the neural candidate either clears the primary margin outright,
    or sits within the margin while showing a pre-registered *substantive*
    improvement on worst-subject F1 / weakest-class F1 / primary-std reduction.
    """
    delta = nn.record_macro_f1_mean - svm.record_macro_f1_mean
    if delta > margin:
        return True
    if delta < -margin:
        return False
    improved = (
        nn.worst_subject_macro_f1_mean
        >= svm.worst_subject_macro_f1_mean + substantial_worst_subject_f1
    )
    improved = improved or (
        nn.weakest_class_record_f1 >= svm.weakest_class_record_f1 + substantial_weakest_class_f1
    )
    improved = improved or (
        svm.record_macro_f1_std - nn.record_macro_f1_std >= substantial_std_reduction
    )
    return improved


def select_full_winner(
    candidates: Sequence[FullCandidateResult],
    *,
    margin: float = SELECTION_MARGIN,
    substantial_worst_subject_f1: float = SUBSTANTIAL_WORST_SUBJECT_F1,
    substantial_weakest_class_f1: float = SUBSTANTIAL_WEAKEST_CLASS_F1,
    substantial_std_reduction: float = SUBSTANTIAL_STD_REDUCTION,
) -> str:
    """Pick the winning candidate from the frozen P5.2-C selection rule.

    Order of application (section 七): (1) gate/evidence failures are excluded;
    (2) primary metric is ``record_macro_f1_mean``; (3) near-tie within ``margin``;
    (4) tie-break ladder ``record_balanced_acc_mean`` then
    ``worst_subject_macro_f1_mean``; (5) within a near-tie the SVM is preferred
    unless a neural candidate shows a pre-registered substantive improvement;
    (6)/(7) calibration/params/inference/training never alter ranking, and no
    freeze happens here. Returns the winning model name.
    """
    eligible = [c for c in candidates if c.passed_gate]
    if not eligible:
        raise ValueError("No candidates passed the gate/evidence check.")

    svms = [c for c in candidates if c.is_frozen_svm]
    if len(svms) != 1:
        raise ValueError("Exactly one frozen SVM reference is required.")
    svm = svms[0]
    if not svm.passed_gate:
        raise ValueError("Frozen SVM reference failed the gate; cannot rank.")

    top_primary = max(c.record_macro_f1_mean for c in eligible)
    near_tie = [c for c in eligible if top_primary - c.record_macro_f1_mean <= margin]

    survivors = [
        c
        for c in near_tie
        if c.is_frozen_svm
        or _neural_beats_svm(
            c,
            svm,
            margin=margin,
            substantial_worst_subject_f1=substantial_worst_subject_f1,
            substantial_weakest_class_f1=substantial_weakest_class_f1,
            substantial_std_reduction=substantial_std_reduction,
        )
    ]
    if not survivors:
        raise ValueError("No candidate survives the near-tie SVM-preference rule.")

    winner = max(
        survivors,
        key=lambda c: (
            c.record_balanced_acc_mean,
            c.worst_subject_macro_f1_mean,
            1 if c.is_frozen_svm else 0,
        ),
    )
    return winner.model
