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

import math
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
# Frozen training parameters (identical for all three NN candidates)
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0
LOSS = "cross_entropy"
OPTIMIZER = "AdamW"
DETERMINISTIC_CUDNN = True
CUDNN_BENCHMARK = False
#: AMP path: current torch autocast + GradScaler (not legacy apex / torch.cuda.amp).
AMP_STRATEGY = "torch_autocast_gradscaler"
#: Frozen 5-class label order for the PoPu fixed-posture task.
FROZEN_LABEL_ORDER: tuple[str, ...] = ("empty", "supine", "prone", "left", "right")

# ---------------------------------------------------------------------------
# Deterministic per-fold training seeds (Stage A / Stage B)
# ---------------------------------------------------------------------------

#: Shared stride between the inner seed and both training seeds (same value as
#: ``full_splits.INNER_SEED_STRIDE``); kept here so the formula is auditable.
TRAIN_SEED_STRIDE = 100
STAGE_A_TRAIN_SEED_BASE = 2_000_000
STAGE_B_REFIT_SEED_BASE = 3_000_000
#: Frozen formula strings, cross-checked against the config. Deterministic
#: arithmetic only — no Python ``hash()``, no process-global random state.
STAGE_A_TRAIN_SEED_FORMULA = "2_000_000 + outer_seed * 100 + local_fold"
STAGE_B_REFIT_SEED_FORMULA = "3_000_000 + outer_seed * 100 + local_fold"

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

#: Final fixed complexity priority, applied ONLY when the primary metric,
#: balanced accuracy and worst-subject F1 are all tied. Prefers the frozen SVM
#: reference first, then smaller networks (tiny_cnn < small_resnet < matrix_mlp).
COMPLEXITY_PRIORITY: tuple[str, ...] = (
    "calibrated_linear_svm",
    "tiny_cnn",
    "small_resnet",
    "matrix_mlp",
)

# ---------------------------------------------------------------------------
# Frozen SVM reference artifacts (P5.1 evidence) — bound by path + SHA-256 + size
# ---------------------------------------------------------------------------

#: The six frozen P5.1 files carrying the ``calibrated_linear_svm`` evidence.
#: The Full runner must verify each SHA-256 before reading and use only the
#: ``calibrated_linear_svm`` rows. Recorded here read-only; never modified.
FROZEN_SVM_REFERENCE_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "path": "data/processed/popu/popu_model_comparison_p5_1_oof_predictions_v0.1.csv",
        "sha256": "807afca919b7737964b870fc4a521a2b1e79bacb669b9eb4f6027b404c6f1ecb",
        "size_bytes": 191504818,
    },
    {
        "path": "outputs/metrics/popu_model_comparison_p5_1_record_level_v0.1.csv",
        "sha256": "13aafaaf048ba4c412e6aaf51f002208bbfa40dd496d218b84ed0d9bd595e798",
        "size_bytes": 10798258,
    },
    {
        "path": "outputs/metrics/popu_model_comparison_p5_1_summary_v0.1.csv",
        "sha256": "8a637809f1d29ea7f885459b49406ac8fe5f87898328792ca9bffe7ba6b00366",
        "size_bytes": 2281,
    },
    {
        "path": "outputs/metrics/popu_model_comparison_p5_1_fold_repeat_v0.1.csv",
        "sha256": "a7c34ca17ec2384320ed2c87637786b7fa408316e3d4fe50b8085cf7afedf986",
        "size_bytes": 7375,
    },
    {
        "path": "outputs/metrics/popu_model_comparison_p5_1_per_class_v0.1.csv",
        "sha256": "7ec36b48ffadc3f96f989791b769cb6ed4f1360543613f30bfea90316bbe4291",
        "size_bytes": 5349,
    },
    {
        "path": "outputs/metrics/popu_model_comparison_p5_1_per_subject_v0.1.csv",
        "sha256": "3b1f757ee17feb6c4346060234bd5817467ffc3339a345802ccae06eee6f7d13",
        "size_bytes": 49669,
    },
)

# ---------------------------------------------------------------------------
# Calibration diagnostics (frozen formulas; diagnostic only, never ranking)
# ---------------------------------------------------------------------------

CALIBRATION_NLL_CLIP = 1e-15
CALIBRATION_NLL_FORMULA = "-mean(log(clip(p_true, 1e-15, 1)))"
CALIBRATION_BRIER_FORMULA = "mean(sum_k((p_k - y_k)^2))"
CALIBRATION_ECE_CONFIDENCE = "max probability"
CALIBRATION_ECE_CORRECT = "argmax == label"
CALIBRATION_ECE_BINS = 15
CALIBRATION_ECE_BINNING = "left-closed right-open; last bin includes 1.0"
CALIBRATION_ECE_WEIGHTING = (
    "weighted by bin sample proportion of abs(accuracy - confidence)"
)
CALIBRATION_ROW_SUM_TOLERANCE = 1e-6


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


def _require_finite_in_range(
    value: float, label: str, *, lo: float = 0.0, hi: float = 1.0
) -> float:
    """Fail closed when a ranking metric is non-finite or outside ``[lo, hi]``.

    NaN/Inf immediately raise; this runs before any ranking so a poisoned metric
    can never silently win (or lose) a comparison.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Full protocol violation: {label} is not numeric ({value!r}).")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(f"Full protocol violation: {label} is non-finite ({value!r}).")
    if not lo <= fvalue <= hi:
        raise ValueError(
            f"Full protocol violation: {label}={fvalue!r} outside reasonable "
            f"range [{lo}, {hi}]."
        )
    return fvalue


def _sorted_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, int]]:
    """Canonical ``(path, sha256, size_bytes)`` tuples, sorted by path."""
    return sorted(
        (
            str(artifact["path"]),
            str(artifact["sha256"]).lower(),
            int(artifact["size_bytes"]),
        )
        for artifact in artifacts
    )


def _complexity_priority_score(
    model: str, priority: Sequence[str] = COMPLEXITY_PRIORITY
) -> int:
    """Return a higher-is-better score from the fixed complexity priority.

    The frozen order is ``calibrated_linear_svm`` → ``tiny_cnn`` →
    ``small_resnet`` → ``matrix_mlp`` (lower complexity preferred). Unknown
    models score ``0`` so they never win a complexity tie.
    """
    try:
        index = list(priority).index(model)
    except ValueError:
        return 0
    return len(priority) - index


def validate_calibration_probabilities(
    probs: Sequence[Sequence[float]], labels: Sequence[int]
) -> None:
    """Fail closed on malformed record-level probability rows.

    Every probability must be finite and in ``[0, 1]``, and each row must sum to
    1 within ``CALIBRATION_ROW_SUM_TOLERANCE``. Also checks ``len(probs) ==
    len(labels)`` and that every label is an integer class index inside its row.
    """
    if len(probs) != len(labels):
        raise ValueError(
            f"Full protocol violation: {len(probs)} prob rows vs {len(labels)} labels."
        )
    for index, (row, label) in enumerate(zip(probs, labels)):
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"Full protocol violation: label at row {index} is not an integer."
            )
        if not 0 <= label < len(row):
            raise ValueError(
                f"Full protocol violation: label {label} out of range for row {index}."
            )
        row_sum = 0.0
        for value in row:
            fvalue = float(value)
            if not math.isfinite(fvalue):
                raise ValueError(
                    f"Full protocol violation: row {index} has a non-finite probability."
                )
            if not 0.0 <= fvalue <= 1.0:
                raise ValueError(
                    f"Full protocol violation: row {index} probability "
                    f"{fvalue!r} outside [0, 1]."
                )
            row_sum += fvalue
        if abs(row_sum - 1.0) > CALIBRATION_ROW_SUM_TOLERANCE:
            raise ValueError(
                f"Full protocol violation: row {index} sums to {row_sum!r}, "
                f"not 1.0 within {CALIBRATION_ROW_SUM_TOLERANCE}."
            )


def record_multiclass_nll(
    probs: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """Record-level multiclass negative log-likelihood (frozen formula).

    ``NLL = -mean(log(clip(p_true, 1e-15, 1)))`` where ``p_true`` is the
    probability assigned to each record's true class.
    """
    validate_calibration_probabilities(probs, labels)
    total = 0.0
    for row, label in zip(probs, labels):
        p_true = min(max(float(row[label]), CALIBRATION_NLL_CLIP), 1.0)
        total += -math.log(p_true)
    return total / len(probs)


def record_multiclass_brier(
    probs: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """Record-level multiclass Brier score (frozen formula).

    ``Brier = mean(sum_k((p_k - y_k)^2))`` where ``y_k`` is the one-hot target.
    """
    validate_calibration_probabilities(probs, labels)
    total = 0.0
    for row, label in zip(probs, labels):
        row_brier = 0.0
        for k, p_k in enumerate(row):
            y_k = 1.0 if k == label else 0.0
            row_brier += (float(p_k) - y_k) ** 2
        total += row_brier
    return total / len(probs)


def record_ece(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_bins: int = CALIBRATION_ECE_BINS,
) -> float:
    """Record-level expected calibration error (frozen formula).

    ``confidence = max(p)``, ``correct = (argmax(p) == label)``. Bins are
    left-closed right-open over ``[0, 1]``; the final bin also includes 1.0.
    ECE is the sample-proportion-weighted mean of ``abs(accuracy - confidence)``
    per bin.
    """
    validate_calibration_probabilities(probs, labels)
    counts = [0] * n_bins
    correct_sums = [0.0] * n_bins
    confidence_sums = [0.0] * n_bins
    for row, label in zip(probs, labels):
        row = [float(value) for value in row]
        confidence = max(row)
        prediction = row.index(confidence)
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        counts[bin_index] += 1
        confidence_sums[bin_index] += confidence
        if prediction == label:
            correct_sums[bin_index] += 1.0
    ece = 0.0
    for index in range(n_bins):
        if counts[index] == 0:
            continue
        accuracy = correct_sums[index] / counts[index]
        avg_confidence = confidence_sums[index] / counts[index]
        ece += (counts[index] / len(probs)) * abs(accuracy - avg_confidence)
    return ece


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
    _require_equal(optimizer.get("name"), OPTIMIZER, "inner_epoch_selection.optimizer.name")
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
    _require_equal(
        tuple(selection.get("complexity_priority", [])),
        COMPLEXITY_PRIORITY,
        "model_selection.complexity_priority",
    )

    # Frozen training parameters (identical across the three NN candidates).
    training = params.get("training_params")
    if not isinstance(training, Mapping):
        raise ValueError("Full config parameters.training_params is required.")
    _require_equal(int(training.get("batch_size")), BATCH_SIZE, "training_params.batch_size")
    _require_equal(int(training.get("num_workers")), NUM_WORKERS, "training_params.num_workers")
    _require_equal(training.get("loss"), LOSS, "training_params.loss")
    _require_equal(training.get("optimizer"), OPTIMIZER, "training_params.optimizer")
    _require_equal(
        bool(training.get("deterministic_cudnn")),
        DETERMINISTIC_CUDNN,
        "training_params.deterministic_cudnn",
    )
    _require_equal(
        bool(training.get("cudnn_benchmark")),
        CUDNN_BENCHMARK,
        "training_params.cudnn_benchmark",
    )
    _require_equal(training.get("amp"), AMP_STRATEGY, "training_params.amp")
    _require_equal(
        tuple(training.get("frozen_label_order", [])),
        FROZEN_LABEL_ORDER,
        "training_params.frozen_label_order",
    )

    # Deterministic per-fold training seeds (Stage A / Stage B).
    seeds = params.get("training_seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("Full config parameters.training_seeds is required.")
    _require_equal(
        seeds.get("stage_a_train_seed"),
        STAGE_A_TRAIN_SEED_FORMULA,
        "training_seeds.stage_a_train_seed",
    )
    _require_equal(
        seeds.get("stage_b_refit_seed"),
        STAGE_B_REFIT_SEED_FORMULA,
        "training_seeds.stage_b_refit_seed",
    )
    _require_equal(
        bool(seeds.get("reset_seed_before_each_candidate")),
        True,
        "training_seeds.reset_seed_before_each_candidate",
    )
    _require_equal(
        bool(seeds.get("all_candidates_share_same_derived_seed")),
        True,
        "training_seeds.all_candidates_share_same_derived_seed",
    )

    # Frozen SVM reference artifacts (P5.1 evidence pinned by path+SHA-256+size).
    artifacts = params.get("frozen_svm_reference_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Full config parameters.frozen_svm_reference_artifacts is required.")
    _require_equal(
        _sorted_artifacts(artifacts),
        _sorted_artifacts(FROZEN_SVM_REFERENCE_ARTIFACTS),
        "frozen_svm_reference_artifacts",
    )

    # Calibration diagnostics (frozen formulas; diagnostic only, never ranking).
    calibration = params.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Full config parameters.calibration is required.")
    _require_equal(calibration.get("record_nll"), CALIBRATION_NLL_FORMULA, "calibration.record_nll")
    _require_equal(
        calibration.get("record_brier"),
        CALIBRATION_BRIER_FORMULA,
        "calibration.record_brier",
    )
    _require_equal(int(calibration.get("ece_bins")), CALIBRATION_ECE_BINS, "calibration.ece_bins")
    _require_equal(
        float(calibration.get("row_sum_tolerance")),
        CALIBRATION_ROW_SUM_TOLERANCE,
        "calibration.row_sum_tolerance",
    )
    _require_equal(
        bool(calibration.get("diagnostic_only_not_ranking")),
        True,
        "calibration.diagnostic_only_not_ranking",
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

    Order of application (section 七): (0) all ranking metrics must be finite and
    in range (NaN/Inf fails closed); (1) gate/evidence failures are excluded;
    (2) primary metric is ``record_macro_f1_mean``; (3) near-tie within ``margin``;
    (4) tie-break ladder ``record_balanced_acc_mean`` then
    ``worst_subject_macro_f1_mean`` then the fixed ``complexity_priority``;
    (5) within a near-tie the SVM is preferred unless a neural candidate shows a
    pre-registered substantive improvement; (6)/(7) calibration/params/inference/
    training never alter ranking, and no freeze happens here. Returns the winning
    model name. The selection is order-independent: swapping candidate input
    order yields the same winner.
    """
    # Ranking metrics must be finite and in a reasonable range before any
    # ranking; NaN/Inf fails closed so a poisoned metric cannot silently win.
    for candidate in candidates:
        _require_finite_in_range(
            candidate.record_macro_f1_mean, f"{candidate.model}.record_macro_f1_mean"
        )
        _require_finite_in_range(
            candidate.record_balanced_acc_mean, f"{candidate.model}.record_balanced_acc_mean"
        )
        _require_finite_in_range(
            candidate.worst_subject_macro_f1_mean,
            f"{candidate.model}.worst_subject_macro_f1_mean",
        )
        _require_finite_in_range(
            candidate.weakest_class_record_f1, f"{candidate.model}.weakest_class_record_f1"
        )
        _require_finite_in_range(
            candidate.record_macro_f1_std, f"{candidate.model}.record_macro_f1_std"
        )

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
            _complexity_priority_score(c.model),
        ),
    )
    return winner.model
