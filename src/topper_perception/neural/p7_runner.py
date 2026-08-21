"""P7 PoPu software-robustness evaluation runner (Reviewer-revised).

This module reads the frozen P5.2-C Full evidence pack (15 ``small_resnet``
``stage_b_final.pt`` checkpoints across 3 repeats × 5 folds), re-loads the
corresponding outer-test raw pressure matrices for each fold, and re-runs
clean inference *plus* every frozen software-perturbation condition on top of
the same frozen checkpoints.

Reviewer-mandated revisions in v0.1.1:

1. P6 single-checkpoint threshold is loaded from the frozen P6 evidence JSON
   (``EXP-P6-POPU-REJECT-20260820-R01/summary.json``) with SHA-256 verification.
   No threshold is hardcoded in this runner.
2. P6.1 ensemble rule is loaded from its own evidence JSON
   (``EXP-P6.1-POPU-CALIBRATION-20260820-R01/summary.json``), produces its own
   metrics, and is reported alongside — not as a replacement for — the P6
   single-checkpoint baseline.
3. Full P7 metrics are computed on a *stitched* OOF across all 15 folds; the
   runner never averages per-fold metrics.
4. Outputs include per-class, per-subject, worst subject, error cases, and
   delta-vs-clean mean / std / worst per condition.
5. The frozen P7 config pins every seed and every perturbation value, and the
   full path requires 3 repeats × 5 folds (= 15) before any sweep begins.
6. The runner fail-closes on missing or inconsistent SHA-256 between the
   evidence pack files (``complete.json``, ``stage_b_final.pt``,
   ``split_manifest.json``) and any drift in their pinned SHA values.
7. A separate ``--clean-only-full-fold`` mode re-infers all outer-test records
   of one fold on CPU and asserts the OOF cross-check is exhaustive.

The runner is governed: any drift in the frozen P5.2-C protocol contract,
checkpoint SHA-256, P7 configuration, or P6/P6.1 evidence SHA fails closed.
The CPU Smoke path restricts itself to one repeat, one fold, and a small
record subset so it can be exercised locally without touching the GPU
evidence pack contents.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from topper_perception.experiments.artifacts import atomic_write_json, sha256_hex
from topper_perception.neural.checkpoint import (
    load_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.data import (
    FROZEN_LABELS,
    LABEL_TO_INDEX,
    MatrixNormalizer,
)
from topper_perception.neural.dataset import PressureDataset, build_dataloader
from topper_perception.neural.full import (
    DATA_BOUNDARY,
    PROBA_COLUMNS,
    RECORD_COLUMNS,
    SNAPSHOT_COLUMNS,
    FullCohort,
    aggregate_record_rows,
    load_full_cohort,
)
from topper_perception.neural.metrics import compute_classification_metrics
from topper_perception.neural.models import build_model
from topper_perception.neural.p6_1 import (
    aggregate_repeat_ensemble,
    calibrated_frame,
    per_subject_metrics as p6_1_per_subject_metrics,
    selective_metrics as p6_1_selective_metrics,
    temperature_scale,
)
from topper_perception.neural.p6_evidence import (
    P61EnsembleRule,
    P6SingleRule,
    load_p6_1_ensemble_rule,
    load_p6_single_rule,
)
from topper_perception.neural.p6_reject import (
    PROBA_COLUMNS as P6_PROBA_COLUMNS,
    RejectRule,
    add_uncertainty_columns,
    apply_rule,
    confusion_matrix,
    error_cases,
    grouped_metrics,
    threshold_metrics,
)
from topper_perception.neural.p7_robustness import (
    add_relative_gaussian_noise,
    downsample_nearest,
    inject_bad_cells,
    inject_bad_lines,
)
from topper_perception.neural.training import predict, resolve_device

# ---------------------------------------------------------------------------
# Frozen contracts (mirrors configs/analysis/popu_p7_robustness_v0.1.json)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "p7-robustness-v0.1"
MODEL_FAMILY = "small_resnet"
LEVEL = "record"
PROTOCOL_NAME = "popu_neural_full_v0.1"  # The evidence pack is frozen P5.2-C Full.
SNAPSHOTS_PER_RECORD = int(DATA_BOUNDARY["snapshots_per_record"])
PROBA_COLUMNS = P6_PROBA_COLUMNS

#: Filename stem of the frozen P5.2-C Full evidence pack.
EVIDENCE_EXP_ID = "EXP-P5.2-C-FULL-COMPARISON-20260820-R01"
#: Local evidence-pack root after extraction. The runner never assumes a
#: ``/root/autodl-tmp`` path; the caller supplies an absolute path.
DEFAULT_EVIDENCE_ROOT = Path(
    r"C:\Users\23939\AppData\Local\Temp\smarttopper-autodl"
    r"\p7-extract\outputs\experiments"
    rf"\{EVIDENCE_EXP_ID}"
)

#: Mandatory fold count for the Full path. Smoke mode is allowed to relax this
#: for early failure-mode discovery but never for Full P7 results.
FULL_REPEATS: tuple[int, ...] = (0, 1, 2)
FULL_LOCAL_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
FULL_TOTAL_FOLDS = len(FULL_REPEATS) * len(FULL_LOCAL_FOLDS)


# ---------------------------------------------------------------------------
# Frozen-condition parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class P7Condition:
    """One deterministic software perturbation condition.

    ``name`` is the canonical CSV/JSON key (lowercase, deterministic).
    ``kind`` selects the underlying :mod:`p7_robustness` function.
    """

    name: str
    kind: str
    params: tuple[tuple[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
        }


def _validate_p7_config(config: Mapping[str, Any]) -> None:
    """Fail closed on any drift in the frozen P7 config."""
    if not isinstance(config, Mapping):
        raise ValueError("P7 config must be a mapping.")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"P7 config schema_version must be {SCHEMA_VERSION!r}, "
            f"got {config.get('schema_version')!r}."
        )
    if config.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"P7 config model_family must be {MODEL_FAMILY!r}, got {config.get('model_family')!r}."
        )
    if config.get("level") != LEVEL:
        raise ValueError(
            f"P7 config level must be {LEVEL!r}, got {config.get('level')!r}."
        )
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not all(isinstance(item, int) for item in seeds):
        raise ValueError("P7 config seeds must be a list of integers.")
    if not seeds:
        raise ValueError("P7 config seeds must be non-empty.")
    if set(seeds) != {701, 702, 703, 704, 705}:
        raise ValueError(
            f"P7 config seeds must be exactly [701, 702, 703, 704, 705]; got {sorted(seeds)}."
        )
    repeats = config.get("repeats")
    if not isinstance(repeats, list) or not all(isinstance(item, int) for item in repeats):
        raise ValueError("P7 config repeats must be a list of integers.")
    if set(repeats) != set(FULL_REPEATS):
        raise ValueError(
            f"P7 config repeats must be exactly {list(FULL_REPEATS)}; got {sorted(repeats)}."
        )
    local_folds = config.get("local_folds")
    if not isinstance(local_folds, list) or not all(isinstance(item, int) for item in local_folds):
        raise ValueError("P7 config local_folds must be a list of integers.")
    if set(local_folds) != set(FULL_LOCAL_FOLDS):
        raise ValueError(
            f"P7 config local_folds must be exactly {list(FULL_LOCAL_FOLDS)}; got {sorted(local_folds)}."
        )
    conditions = config.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("P7 config conditions must be a mapping.")
    expected = {
        "density_nearest",
        "gaussian_noise_p95_fraction",
        "bad_cell_fraction",
        "bad_rows",
        "bad_columns",
    }
    if set(conditions.keys()) != expected:
        raise ValueError(
            "P7 config conditions keys must equal the frozen set "
            f"{sorted(expected)}; got {sorted(conditions.keys())}."
        )
    for key, values in conditions.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"P7 config conditions.{key} must be a non-empty list.")
    if not isinstance(config.get("p6_evidence"), Mapping):
        raise ValueError("P7 config p6_evidence block is required.")
    if str(config["p6_evidence"].get("single_threshold_source", {}).get("kind")) != "summary_json":
        raise ValueError("P7 p6_evidence.single_threshold_source.kind must be 'summary_json'.")
    if str(config["p6_evidence"].get("ensemble_rule_source", {}).get("kind")) != "summary_json":
        raise ValueError("P7 p6_evidence.ensemble_rule_source.kind must be 'summary_json'.")
    if str(config.get("stitching", {}).get("policy")) != "pool_first_then_metric":
        raise ValueError(
            "P7 config stitching.policy must be 'pool_first_then_metric'."
        )


def parse_p7_conditions(config: Mapping[str, Any]) -> list[P7Condition]:
    """Expand the frozen P7 condition tree into a deterministic flat list."""
    _validate_p7_config(config)
    conditions: list[P7Condition] = []

    for spec in config["conditions"]["density_nearest"]:
        if not isinstance(spec, Mapping):
            raise ValueError("density_nearest entries must be mappings.")
        row_stride = int(spec["row_stride"])
        column_stride = int(spec["column_stride"])
        conditions.append(
            P7Condition(
                name=f"density_stride_{row_stride}_{column_stride}",
                kind="density_nearest",
                params=(("row_stride", row_stride), ("column_stride", column_stride)),
            )
        )

    for fraction in config["conditions"]["gaussian_noise_p95_fraction"]:
        fraction = float(fraction)
        conditions.append(
            P7Condition(
                name=f"noise_p95_{fraction:.2f}",
                kind="gaussian_noise",
                params=(("sigma_fraction", fraction),),
            )
        )

    for fraction in config["conditions"]["bad_cell_fraction"]:
        fraction = float(fraction)
        conditions.append(
            P7Condition(
                name=f"bad_cell_{fraction:.2f}",
                kind="bad_cell",
                params=(("fraction", fraction),),
            )
        )

    for count in config["conditions"]["bad_rows"]:
        count = int(count)
        conditions.append(
            P7Condition(
                name=f"bad_rows_{count}",
                kind="bad_lines",
                params=(("bad_rows", count), ("bad_columns", 0)),
            )
        )

    for count in config["conditions"]["bad_columns"]:
        count = int(count)
        conditions.append(
            P7Condition(
                name=f"bad_columns_{count}",
                kind="bad_lines",
                params=(("bad_rows", 0), ("bad_columns", count)),
            )
        )

    return conditions


# ---------------------------------------------------------------------------
# Evidence-pack discovery and integrity (Reviewer point 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldCheckpoint:
    """Resolved reference to one fold's frozen ``stage_b_final.pt``.

    All SHA-256 fields are pinned against the on-disk bytes; any drift raises
    :class:`ValueError` during construction.
    """

    repeat: int
    local_fold: int
    outer_seed: int
    checkpoint_path: Path
    summary_path: Path
    complete_path: Path
    record_predictions_path: Path
    split_manifest_sha256: str
    split_manifest_actual_sha256: str
    expected_size_bytes: int
    expected_sha256: str
    complete_sha256: str
    outer_train_subjects: tuple[str, ...]
    outer_test_subjects: tuple[str, ...]
    model_config: dict[str, Any] = field(default_factory=dict)


def resolve_evidence_root(evidence_root: Path | str | None) -> Path:
    """Return the absolute evidence-pack root, defaulting to the known path."""
    if evidence_root is None:
        root = DEFAULT_EVIDENCE_ROOT
    else:
        root = Path(evidence_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"P7 evidence-pack root not found: {root}. Extract "
            f"{EVIDENCE_EXP_ID} before invoking the runner."
        )
    return root


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """The frozen split manifest verified via canonical-JSON SHA only.

    Per Reviewer point #1 the canonical SHA is computed over the manifest
    content with the ``sha256`` field stripped (the same recipe
    :func:`topper_perception.neural.full_splits._canonical_sha256` and
    :func:`topper_perception.neural.full_splits.validate_full_fold_manifest`
    use). The original file byte SHA is reported as an extra file hash but
    **never** compared against the declared canonical hash: a file rewritten
    with different whitespace or key ordering will keep the canonical digest
    but change the byte digest, which is the intended semantics.
    """

    path: Path
    protocol: str
    n_folds: int
    declared_canonical_sha256: str | None
    canonical_sha256: str
    file_byte_sha256: str
    folds: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.declared_canonical_sha256 is None:
            raise ValueError(
                f"split_manifest.json at {self.path} has no declared "
                "'sha256' field; canonical verification is mandatory."
            )
        if self.declared_canonical_sha256.lower() != self.canonical_sha256.lower():
            raise ValueError(
                f"split_manifest.json declared canonical sha256 "
                f"{self.declared_canonical_sha256!r} does not match recomputed "
                f"canonical sha256 {self.canonical_sha256!r} at {self.path}."
            )


def load_split_manifest(evidence_root: Path) -> SplitManifest:
    """Load the frozen split manifest with fail-closed canonical SHA verification."""
    from topper_perception.neural.full_splits import _canonical_sha256

    path = evidence_root / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"split_manifest.json not found under {evidence_root}.")
    raw_bytes = path.read_bytes()
    file_byte_sha = sha256_hex(path)
    manifest = json.loads(raw_bytes)
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise ValueError(
            f"Split manifest protocol must be {PROTOCOL_NAME!r}, "
            f"got {manifest.get('protocol')!r}."
        )
    folds = tuple(manifest.get("folds", []))
    if not folds:
        raise ValueError("Split manifest contains no folds.")
    content_without_sha = {k: v for k, v in manifest.items() if k != "sha256"}
    canonical_sha = _canonical_sha256(content_without_sha)
    return SplitManifest(
        path=path,
        protocol=str(manifest.get("protocol")),
        n_folds=len(folds),
        declared_canonical_sha256=(
            str(manifest["sha256"]).lower() if "sha256" in manifest else None
        ),
        canonical_sha256=canonical_sha,
        file_byte_sha256=file_byte_sha,
        folds=folds,
    )


def _resolve_complete_json_artifacts(complete: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index complete.json artifacts by their trailing filename."""
    indexed: dict[str, dict[str, Any]] = {}
    for artifact in complete.get("artifacts", []):
        path = str(artifact.get("path", ""))
        if not path:
            continue
        indexed[path.rsplit("/", 1)[-1]] = dict(artifact)
    return indexed


def resolve_fold_checkpoints(
    evidence_root: Path,
    *,
    model_family: str = MODEL_FAMILY,
    repeats: Sequence[int] | None = None,
    local_folds: Sequence[int] | None = None,
    require_full: bool = False,
) -> list[FoldCheckpoint]:
    """Resolve every requested fold's frozen checkpoint reference.

    Every fold is fail-closed against:

    - the on-disk SHA-256 of its ``stage_b_final.pt``;
    - the on-disk SHA-256 of ``complete.json``'s ``stage_b_final.pt`` entry;
    - the on-disk SHA-256 of ``split_manifest.json`` vs. its declared
      ``sha256`` and the per-fold summary's ``split_manifest_sha256`` field.

    When ``require_full`` is true (the Full P7 path), the runner rejects any
    evidence pack that does not provide all ``FULL_REPEATS × FULL_LOCAL_FOLDS``
    (= 15) checkpoints even if the caller requested a subset.
    """
    manifest = load_split_manifest(evidence_root)

    requested_repeats: set[int]
    requested_folds: set[int]
    if require_full:
        requested_repeats = set(FULL_REPEATS)
        requested_folds = set(FULL_LOCAL_FOLDS)
    else:
        requested_repeats = set(repeats) if repeats is not None else set()
        requested_folds = set(local_folds) if local_folds is not None else set()
        if repeats is not None:
            requested_repeats = {int(item) for item in repeats}
        if local_folds is not None:
            requested_folds = {int(item) for item in local_folds}

    selected: list[FoldCheckpoint] = []
    seen_keys: set[tuple[int, int]] = set()
    for fold in manifest.folds:
        repeat = int(fold["repeat"])
        local_fold = int(fold["local_fold"])
        if requested_repeats and repeat not in requested_repeats:
            continue
        if requested_folds and local_fold not in requested_folds:
            continue
        key = (repeat, local_fold)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate fold in evidence pack: repeat={repeat}, local_fold={local_fold}."
            )
        seen_keys.add(key)

        candidate_dir = (
            evidence_root
            / "folds"
            / f"repeat_{repeat}"
            / f"fold_{local_fold}"
            / model_family
        )
        checkpoint_path = candidate_dir / "stage_b_final.pt"
        summary_path = candidate_dir / "summary.json"
        complete_path = candidate_dir / "complete.json"
        record_predictions_path = candidate_dir / "record_predictions.csv"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for repeat={repeat} local_fold={local_fold}: "
                f"{checkpoint_path}"
            )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary for fold: {summary_path}")
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"Missing complete.json for repeat={repeat} local_fold={local_fold}: "
                f"{complete_path}"
            )
        if not record_predictions_path.is_file():
            raise FileNotFoundError(
                f"Missing OOF record_predictions.csv for fold: {record_predictions_path}"
            )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("model") != model_family:
            raise ValueError(
                f"summary.json model mismatch at repeat={repeat} local_fold={local_fold}: "
                f"expected {model_family!r}, got {summary.get('model')!r}."
            )
        if int(summary["repeat"]) != repeat or int(summary["local_fold"]) != local_fold:
            raise ValueError(
                f"summary.json repeat/local_fold mismatch at repeat={repeat} "
                f"local_fold={local_fold}."
            )
        if str(summary.get("split_manifest_sha256", "")).lower() != manifest.canonical_sha256.lower():
            raise ValueError(
                f"summary.json split_manifest_sha256 mismatch at repeat={repeat} "
                f"local_fold={local_fold}: summary={summary.get('split_manifest_sha256')!r}, "
                f"on-disk={manifest.canonical_sha256!r}."
            )
        stage_b = summary["stage_b"]

        checkpoint_sha = sha256_hex(checkpoint_path)
        checkpoint_size = checkpoint_path.stat().st_size
        expected_size = int(stage_b.get("checkpoint_size_bytes", -1))
        if expected_size > 0 and checkpoint_size != expected_size:
            raise ValueError(
                f"Checkpoint size mismatch at repeat={repeat} local_fold={local_fold}: "
                f"on disk={checkpoint_size}, summary={expected_size}."
            )

        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        indexed = _resolve_complete_json_artifacts(complete)
        if "stage_b_final.pt" not in indexed:
            raise ValueError(
                f"complete.json has no stage_b_final.pt artifact for "
                f"repeat={repeat} local_fold={local_fold}."
            )
        marker = indexed["stage_b_final.pt"]
        marker_sha = str(marker["sha256"]).lower()
        marker_size = int(marker["size_bytes"])
        if marker_size != checkpoint_size:
            raise ValueError(
                f"complete.json size mismatch at repeat={repeat} "
                f"local_fold={local_fold}: on disk={checkpoint_size}, "
                f"marker={marker_size}."
            )
        if marker_sha != checkpoint_sha:
            raise ValueError(
                f"complete.json SHA-256 mismatch at repeat={repeat} "
                f"local_fold={local_fold}: on disk={checkpoint_sha}, "
                f"marker={marker_sha}."
            )

        model_config = dict(stage_b.get("model_config") or {})

        selected.append(
            FoldCheckpoint(
                repeat=repeat,
                local_fold=local_fold,
                outer_seed=int(fold.get("outer_seed", 11)),
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                complete_path=complete_path,
                record_predictions_path=record_predictions_path,
                split_manifest_sha256=manifest.canonical_sha256,
                split_manifest_actual_sha256=manifest.canonical_sha256,
                expected_size_bytes=checkpoint_size,
                expected_sha256=checkpoint_sha,
                complete_sha256=sha256_hex(complete_path),
                outer_train_subjects=tuple(fold["outer_train_subjects"]),
                outer_test_subjects=tuple(fold["outer_test_subjects"]),
                model_config=model_config,
            )
        )

    if require_full:
        expected_keys = {(int(r), int(f)) for r in FULL_REPEATS for f in FULL_LOCAL_FOLDS}
        missing = sorted(expected_keys - seen_keys)
        if missing:
            raise ValueError(
                "Full P7 requires exactly "
                f"{FULL_TOTAL_FOLDS} folds (3 repeats × 5 folds); "
                f"missing from evidence pack: {missing}."
            )
        if seen_keys != expected_keys:
            raise ValueError(
                "Full P7 evidence pack contains unexpected folds: "
                f"{sorted(seen_keys - expected_keys)}."
            )
    if not selected:
        raise ValueError(
            "No fold checkpoints matched the requested repeats/local_folds."
        )
    selected.sort(key=lambda item: (item.repeat, item.local_fold))
    return selected


# ---------------------------------------------------------------------------
# Outer-test raw-matrix loading
# ---------------------------------------------------------------------------


def _load_outer_test_samples(
    cohort: FullCohort,
    *,
    outer_test_subjects: Sequence[str],
    record_id_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Restrict the shared Full cohort to one fold's outer-test subjects."""
    test_subjects = set(str(s) for s in outer_test_subjects)
    mask = np.fromiter(
        (subject in test_subjects for subject in cohort.subject_ids),
        dtype=bool,
        count=len(cohort.subject_ids),
    )
    if not mask.any():
        raise ValueError(
            f"No cohort samples found for outer_test_subjects={sorted(test_subjects)}."
        )
    matrices = cohort.matrices[mask]
    labels = cohort.labels[mask]
    sample_ids = [cohort.sample_ids[int(i)] for i in np.flatnonzero(mask)]
    record_ids = [cohort.record_ids[int(i)] for i in np.flatnonzero(mask)]
    subject_ids = [cohort.subject_ids[int(i)] for i in np.flatnonzero(mask)]

    if record_id_filter is not None:
        keep = [rid in record_id_filter for rid in record_ids]
        matrices = matrices[keep]
        labels = labels[keep]
        sample_ids = [s for s, k in zip(sample_ids, keep) if k]
        record_ids = [r for r, k in zip(record_ids, keep) if k]
        subject_ids = [s for s, k in zip(subject_ids, keep) if k]
    if len(record_ids) % SNAPSHOTS_PER_RECORD != 0:
        raise ValueError(
            f"outer-test sample count {len(record_ids)} is not a multiple of "
            f"SNAPSHOTS_PER_RECORD={SNAPSHOTS_PER_RECORD}."
        )
    return matrices, labels, sample_ids, record_ids, subject_ids


def _group_records_by_record(
    sample_ids: Sequence[str],
    record_ids: Sequence[str],
    subject_ids: Sequence[str],
    matrices: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Group per-snapshot rows by record_id into a deterministic ordered list."""
    grouped: dict[str, dict[str, Any]] = {}
    for index in range(len(record_ids)):
        record_id = str(record_ids[index])
        slot = grouped.setdefault(
            record_id,
            {
                "record_id": record_id,
                "subject_id": str(subject_ids[index]),
                "sample_ids": [],
                "matrices": [],
                "labels": [],
            },
        )
        slot["sample_ids"].append(str(sample_ids[index]))
        matrix = matrices[index]
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix.reshape(matrix.shape[1], matrix.shape[2])
        elif matrix.ndim == 4 and matrix.shape[1] == 1:
            matrix = matrix.reshape(matrix.shape[2], matrix.shape[3])
        slot["matrices"].append(matrix)
        slot["labels"].append(int(labels[index]))
    out: list[dict[str, Any]] = []
    for record_id in sorted(grouped):
        slot = grouped[record_id]
        if len(slot["labels"]) != SNAPSHOTS_PER_RECORD:
            raise ValueError(
                f"record {record_id!r} has {len(slot['labels'])} snapshots, "
                f"expected {SNAPSHOTS_PER_RECORD}."
            )
        label_set = set(slot["labels"])
        if len(label_set) != 1:
            raise ValueError(
                f"record {record_id!r} has conflicting labels across snapshots: {sorted(label_set)}."
            )
        out.append(
            {
                "record_id": record_id,
                "subject_id": str(slot["subject_id"]),
                "label": next(iter(label_set)),
                "matrices": np.stack(slot["matrices"]).astype(np.float32),
                "sample_ids": tuple(slot["sample_ids"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Clean and perturbed inference
# ---------------------------------------------------------------------------


def _build_normalizer_from_payload(
    payload: Mapping[str, Any],
    fold_checkpoint: FoldCheckpoint,
) -> MatrixNormalizer:
    normalization = payload.get("normalization") or {}
    mean = normalization.get("mean")
    std = normalization.get("std")
    if mean is None or std is None:
        raise ValueError(
            f"Checkpoint normalization missing for repeat={fold_checkpoint.repeat} "
            f"local_fold={fold_checkpoint.local_fold}."
        )
    normalizer = MatrixNormalizer()
    normalizer.mean_ = float(mean)
    normalizer.std_ = float(std)
    return normalizer


def _load_fold_payload(
    fold_checkpoint: FoldCheckpoint,
    device: torch.device,
) -> tuple[torch.nn.Module, MatrixNormalizer]:
    payload = load_checkpoint(fold_checkpoint.checkpoint_path, map_location=device)
    validate_checkpoint(payload)
    if tuple(payload["frozen_labels"]) != tuple(FROZEN_LABELS):
        raise ValueError(
            f"Checkpoint frozen_labels mismatch at repeat={fold_checkpoint.repeat} "
            f"local_fold={fold_checkpoint.local_fold}."
        )
    model_config = dict(payload.get("model_config") or fold_checkpoint.model_config)
    model = build_model(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    normalizer = _build_normalizer_from_payload(payload, fold_checkpoint)
    return model, normalizer


def _infer_records(
    model: torch.nn.Module,
    normalizer: MatrixNormalizer,
    records: Sequence[dict[str, Any]],
    *,
    fold_checkpoint: FoldCheckpoint | None = None,
    device: torch.device,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run inference on every snapshot of every record and return snapshot rows."""
    sample_ids: list[str] = []
    record_ids: list[str] = []
    subject_ids: list[str] = []
    matrices: list[np.ndarray] = []
    label_lookup: dict[str, int] = {}
    for record in records:
        for snapshot_index, matrix in enumerate(record["matrices"]):
            sample_id = str(record["sample_ids"][snapshot_index])
            sample_ids.append(sample_id)
            record_ids.append(str(record["record_id"]))
            subject_ids.append(str(record["subject_id"]))
            matrices.append(matrix)
        label_lookup[str(record["record_id"])] = int(record["label"])

    if not matrices:
        raise ValueError("Cannot run inference on zero matrices.")

    stacked = np.stack(matrices).astype(np.float32)
    if stacked.ndim != 3 or stacked.shape[1:] != (64, 27):
        raise ValueError(
            f"_infer_records expects (10, 64, 27) per record, got shape "
            f"{stacked.shape} after grouping."
        )
    normalized = normalizer.transform(stacked)
    model_input = np.expand_dims(normalized, axis=1).astype(np.float32, copy=False)
    labels_array = np.asarray(
        [label_lookup[rid] for rid in record_ids], dtype=np.int64
    )

    dataset = PressureDataset(
        model_input,
        labels_array,
        sample_ids=sample_ids,
        record_ids=record_ids,
        subject_ids=subject_ids,
    )
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    result = predict(model, loader, device)

    if not np.isfinite(result.logits).all():
        raise RuntimeError(
            f"Inference produced non-finite logits for records "
            f"{sorted(set(record_ids))[:5]}..."
        )
    if not np.isfinite(result.probabilities).all():
        raise RuntimeError(
            f"Inference produced non-finite probabilities for records "
            f"{sorted(set(record_ids))[:5]}..."
        )
    if bool(((result.probabilities < 0.0) | (result.probabilities > 1.0)).any()):
        raise RuntimeError("Inference probabilities fell outside [0, 1].")
    if not np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Inference probability rows do not sum to 1.")

    repeat_value = int(fold_checkpoint.repeat) if fold_checkpoint is not None else 0
    local_fold_value = int(fold_checkpoint.local_fold) if fold_checkpoint is not None else 0
    outer_seed_value = int(fold_checkpoint.outer_seed) if fold_checkpoint is not None else 11
    snapshot_rows: list[dict[str, Any]] = []
    for index in range(int(result.n_samples)):
        probabilities = result.probabilities[index]
        prediction = int(result.predictions[index])
        row = {
            "model": MODEL_FAMILY,
            "repeat": repeat_value,
            "outer_seed": outer_seed_value,
            "local_fold": local_fold_value,
            "sample_id": str(result.sample_ids[index]),
            "record_id": str(result.record_ids[index]),
            "subject_id": str(result.subject_ids[index]),
            "y_true": FROZEN_LABELS[int(result.labels[index])],
            "y_pred": FROZEN_LABELS[prediction],
            "confidence": float(probabilities[prediction]),
        }
        row.update({col: float(probabilities[i]) for i, col in enumerate(PROBA_COLUMNS)})
        snapshot_rows.append(row)
    return snapshot_rows


def _apply_condition_to_record(
    matrices: np.ndarray,
    condition: P7Condition,
    seed: int,
) -> np.ndarray:
    """Apply one (condition, seed) perturbation to one record's 10-snapshot stack."""
    array = np.asarray(matrices, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (64, 27):
        raise ValueError(
            f"record matrices must have shape (10, 64, 27), got {array.shape}."
        )
    params = dict(condition.params)
    if condition.kind == "density_nearest":
        result = downsample_nearest(
            array, row_stride=params["row_stride"], column_stride=params["column_stride"]
        )
    elif condition.kind == "gaussian_noise":
        result = add_relative_gaussian_noise(
            array, sigma_fraction=params["sigma_fraction"], seed=seed
        )
    elif condition.kind == "bad_cell":
        result, _ = inject_bad_cells(array, fraction=params["fraction"], seed=seed)
    elif condition.kind == "bad_lines":
        result, _ = inject_bad_lines(
            array,
            bad_rows=params["bad_rows"],
            bad_columns=params["bad_columns"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown condition kind: {condition.kind!r}.")
    if result.shape != array.shape:
        raise ValueError(
            f"Perturbation {condition.name!r} changed shape from {array.shape} to {result.shape}."
        )
    if not np.isfinite(result).all():
        raise ValueError(f"Perturbation {condition.name!r} produced non-finite values.")
    if (result < 0).any():
        raise ValueError(f"Perturbation {condition.name!r} produced negative values.")
    return result.astype(np.float32, copy=False)


def derive_record_seed(perturb_seed: int, record_id: str) -> int:
    """Derive a deterministic 64-bit integer seed from ``(perturb_seed, record_id)``.

    Per Round-4 review the Gaussian noise path needs a *per-record* stable
    seed so that, given the same ``(perturb_seed, record_id)``, the noise
    tensor is bit-identical across reruns. Without this derivation every
    record in a single perturbation run would receive the *same* noise
    pattern because the caller only supplies the per-condition / per-seed
    salt. Mixing ``record_id`` into the salt gives every record its own
    independent-but-deterministic noise.

    The derivation uses SHA-256 over a canonical ``"{perturb_seed}|{record_id}"``
    string and takes the first 8 bytes as an unsigned big-endian integer.
    SHA-256 is platform-independent (unlike Python's :func:`hash`) and
    produces uniformly distributed 64-bit values across the integer space,
    which is what :class:`numpy.random.Generator` expects for ``seed``.
    """
    canonical = f"{int(perturb_seed)}|{record_id}".encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def perturb_records(
    records: Sequence[dict[str, Any]],
    condition: P7Condition,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply one (condition, seed) perturbation across every record.

    Per Round-4 review the Gaussian noise path derives a per-record stable
    seed via :func:`derive_record_seed` so different records receive
    different but deterministically reproducible noise patterns. Other
    deterministic conditions (``density_nearest`` is parameter-only;
    ``bad_cell`` and ``bad_lines`` use the caller's seed directly because
    their masking pattern is independent of magnitude) keep the original
    seed argument.
    """
    perturbed: list[dict[str, Any]] = []
    for record in records:
        if condition.kind == "gaussian_noise":
            effective_seed = derive_record_seed(int(seed), str(record["record_id"]))
        else:
            effective_seed = int(seed)
        perturbed.append(
            {
                "record_id": record["record_id"],
                "subject_id": record["subject_id"],
                "label": record["label"],
                "sample_ids": record["sample_ids"],
                "matrices": _apply_condition_to_record(
                    record["matrices"], condition, effective_seed
                ),
            }
        )
    return perturbed


# ---------------------------------------------------------------------------
# Metrics, OOF cross-check, and per-fold summary
# ---------------------------------------------------------------------------


def _record_metric_blocks(record_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true = np.asarray(
        [LABEL_TO_INDEX[str(row["y_true"])] for row in record_rows], dtype=np.int64
    )
    pred = np.asarray(
        [LABEL_TO_INDEX[str(row["y_pred"])] for row in record_rows], dtype=np.int64
    )
    metrics = compute_classification_metrics(true, pred, FROZEN_LABELS)
    return metrics.as_dict()


def _record_p6_single_metrics(
    record_rows: Sequence[Mapping[str, Any]],
    *,
    rule: P6SingleRule,
) -> dict[str, Any]:
    """Apply the frozen P6 single-checkpoint rule to a record set."""
    if not record_rows:
        return {
            "rule_kind": "p6_single",
            "threshold": float(rule.threshold),
            "n": 0,
            "accepted_n": 0,
            "coverage": None,
            "wrong_action_n": 0,
            "wrong_action_rate": None,
            "accepted_accuracy": None,
            "accepted_error_rate": None,
            "source_path": str(rule.source.path),
            "source_sha256": rule.source.actual_sha256,
        }
    frame = pd.DataFrame(list(record_rows))
    frame = add_uncertainty_columns(frame)
    scored = apply_rule(frame, RejectRule(float(rule.threshold)))
    accepted = scored["accepted"]
    n = int(len(scored))
    accepted_n = int(accepted.sum())
    wrong_n = int(scored["wrong_action"].sum())
    return {
        "rule_kind": "p6_single",
        "threshold": float(rule.threshold),
        "n": n,
        "accepted_n": accepted_n,
        "coverage": float(accepted_n) / n if n else None,
        "wrong_action_n": wrong_n,
        "wrong_action_rate": float(wrong_n) / n if n else None,
        "accepted_accuracy": float(scored.loc[accepted, "correct"].mean()) if accepted_n else None,
        "accepted_error_rate": float(wrong_n) / accepted_n if accepted_n else None,
        "source_path": str(rule.source.path),
        "source_sha256": rule.source.actual_sha256,
    }


def _record_p6_1_ensemble_metrics(
    record_rows: Sequence[Mapping[str, Any]],
    *,
    rule: P61EnsembleRule,
) -> dict[str, Any]:
    """Apply the frozen P6.1 ensemble rule to a record set.

    The rule is implemented separately from the P6 single-checkpoint path so
    the uncalibrated single-checkpoint threshold cannot leak into the
    ensemble evaluation. Per Reviewer point #2 the ensemble rule's
    ``threshold`` is the unanimity branch (``rules[1]``) value and never the
    pre-unanimity ``rules[0]`` value. We:

    1. group records by ``(record_id)`` (which is the same record across
       the three repeats of a fold);
    2. require exactly three repeats per record with matching provenance;
    3. temperature-scale the per-repeat probabilities and average them;
    4. accept iff the scaled ensemble top-1 >= threshold AND the three
       repeats unanimously predict the same label.
    """
    if not record_rows:
        return {
            "rule_kind": "p6_1_ensemble",
            "temperature": float(rule.temperature),
            "threshold": float(rule.threshold),
            "require_unanimous": bool(rule.require_unanimous),
            "n": 0,
            "accepted_n": 0,
            "coverage": None,
            "wrong_action_n": 0,
            "wrong_action_rate": None,
            "accepted_accuracy": None,
            "accepted_error_rate": None,
            "source_path": str(rule.source.path),
            "source_sha256": rule.source.actual_sha256,
        }
    frame = pd.DataFrame(list(record_rows))
    try:
        ensemble = aggregate_repeat_ensemble(frame)
    except ValueError as exc:
        # The ensemble cannot be formed (e.g. a perturbed run only has one
        # repeat's worth of rows). Record a structured empty result so the
        # overall report still aggregates cleanly.
        return {
            "rule_kind": "p6_1_ensemble",
            "temperature": float(rule.temperature),
            "threshold": float(rule.threshold),
            "require_unanimous": bool(rule.require_unanimous),
            "n": int(len(frame)),
            "accepted_n": 0,
            "coverage": 0.0,
            "wrong_action_n": 0,
            "wrong_action_rate": 0.0,
            "accepted_accuracy": None,
            "accepted_error_rate": None,
            "source_path": str(rule.source.path),
            "source_sha256": rule.source.actual_sha256,
            "ensemble_error": str(exc),
        }
    calibrated = calibrated_frame(ensemble, temperature=rule.temperature)
    selective = p6_1_selective_metrics(
        calibrated,
        threshold=rule.threshold,
        require_unanimous=rule.require_unanimous,
    )
    return {
        "rule_kind": "p6_1_ensemble",
        "temperature": float(rule.temperature),
        "threshold": float(rule.threshold),
        "require_unanimous": bool(rule.require_unanimous),
        **selective,
        "source_path": str(rule.source.path),
        "source_sha256": rule.source.actual_sha256,
    }


def _assert_clean_matches_oof(
    snapshot_rows: Sequence[Mapping[str, Any]],
    *,
    fold_checkpoint: FoldCheckpoint,
    atol: float = 1e-5,
    exhaustive: bool = False,
) -> dict[str, Any]:
    """Fail closed when the clean re-inference disagrees with the P5.2-C OOF.

    In ``exhaustive`` mode the OOF row count must equal the number of records
    we re-inferred. This is the contract for the 958-record clean-only
    full-fold CPU reproduction.
    """
    record_rows = aggregate_record_rows(snapshot_rows)
    oof = pd.read_csv(fold_checkpoint.record_predictions_path)
    oof_index = {(str(row["record_id"]), int(row["repeat"])): row for _, row in oof.iterrows()}
    compared = 0
    for row in record_rows:
        key = (str(row["record_id"]), int(row["repeat"]))
        if key not in oof_index:
            raise RuntimeError(
                f"Clean OOF missing record at repeat={fold_checkpoint.repeat} "
                f"local_fold={fold_checkpoint.local_fold}: {row['record_id']!r}"
            )
        oof_row = oof_index[key]
        if str(oof_row["y_pred"]) != str(row["y_pred"]):
            raise RuntimeError(
                f"Clean y_pred mismatch on record {row['record_id']!r}: "
                f"re-inferred={row['y_pred']!r}, OOF={oof_row['y_pred']!r}."
            )
        for column in PROBA_COLUMNS:
            if not math.isclose(float(oof_row[column]), float(row[column]), abs_tol=atol, rel_tol=0):
                raise RuntimeError(
                    f"Clean probability mismatch on record {row['record_id']!r} "
                    f"column {column!r}: re-inferred={row[column]}, OOF={oof_row[column]}."
                )
        compared += 1
    if exhaustive and compared != len(record_rows):
        raise RuntimeError(
            f"Clean exhaustive OOF cross-check did not consume all inferred "
            f"records: compared={compared}, total_record_rows={len(record_rows)}."
        )
    if exhaustive and compared != len(oof):
        raise RuntimeError(
            f"Clean exhaustive OOF cross-check did not consume all OOF rows: "
            f"compared={compared}, oof_rows={len(oof)}."
        )
    return {
        "oof_records_compared": compared,
        "oof_records_total": int(len(oof)),
        "oof_argmax_identical": True,
        "oof_probability_abs_tol": atol,
        "exhaustive": bool(exhaustive),
    }


# ---------------------------------------------------------------------------
# Stitching (Reviewer point 3) and full metrics
# ---------------------------------------------------------------------------


def _stitch_full_oof(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Pool every (repeat, local_fold, condition, seed) record row together."""
    if not frames:
        return pd.DataFrame()
    stitched = pd.concat(frames, ignore_index=True, sort=False)
    if stitched.empty:
        return stitched
    required = {"model", "repeat", "outer_seed", "local_fold", "record_id",
                "subject_id", "y_true", "y_pred", "confidence", "n_snapshots",
                *PROBA_COLUMNS}
    missing = required - set(stitched.columns)
    if missing:
        raise ValueError(
            f"Stitched OOF missing required columns: {sorted(missing)}."
        )
    return stitched


def _stitched_classification_metrics(stitched: pd.DataFrame) -> dict[str, Any]:
    """Compute record-level classification metrics on the stitched OOF."""
    if stitched.empty:
        return compute_classification_metrics(
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), FROZEN_LABELS
        ).as_dict()
    true = np.asarray(
        [LABEL_TO_INDEX[str(label)] for label in stitched["y_true"]], dtype=np.int64
    )
    pred = np.asarray(
        [LABEL_TO_INDEX[str(label)] for label in stitched["y_pred"]], dtype=np.int64
    )
    metrics = compute_classification_metrics(true, pred, FROZEN_LABELS)
    return metrics.as_dict()


def _stitched_p6_single(stitched: pd.DataFrame, *, rule: P6SingleRule) -> dict[str, Any]:
    """Apply the P6 single rule to the stitched OOF frame."""
    if stitched.empty:
        return {
            "rule_kind": "p6_single",
            "threshold": float(rule.threshold),
            "n": 0, "accepted_n": 0, "coverage": None,
            "wrong_action_n": 0, "wrong_action_rate": None,
            "accepted_accuracy": None, "accepted_error_rate": None,
            "source_path": str(rule.source.path),
            "source_sha256": rule.source.actual_sha256,
        }
    return _record_p6_single_metrics(stitched.to_dict(orient="records"), rule=rule)


def _stitched_p6_1_ensemble(stitched: pd.DataFrame, *, rule: P61EnsembleRule) -> dict[str, Any]:
    """Apply the P6.1 ensemble rule to the stitched OOF frame."""
    if stitched.empty:
        return {
            "rule_kind": "p6_1_ensemble",
            "temperature": float(rule.temperature),
            "threshold": float(rule.threshold),
            "require_unanimous": bool(rule.require_unanimous),
            "n": 0, "accepted_n": 0, "coverage": None,
            "wrong_action_n": 0, "wrong_action_rate": None,
            "accepted_accuracy": None, "accepted_error_rate": None,
            "source_path": str(rule.source.path),
            "source_sha256": rule.source.actual_sha256,
        }
    return _record_p6_1_ensemble_metrics(stitched.to_dict(orient="records"), rule=rule)


def _per_class_breakdown(
    stitched: pd.DataFrame, *, rule: P6SingleRule
) -> list[dict[str, Any]]:
    if stitched.empty:
        return []
    return _breakdown(stitched, by="y_true", rule=rule)


def _per_subject_breakdown(
    stitched: pd.DataFrame, *, rule: P6SingleRule
) -> list[dict[str, Any]]:
    if stitched.empty:
        return []
    return _breakdown(stitched, by="subject_id", rule=rule)


def _breakdown(
    stitched: pd.DataFrame, *, by: str, rule: P6SingleRule
) -> list[dict[str, Any]]:
    """Compute per-class or per-subject breakdown AFTER applying the P6 single rule.

    Per Reviewer point #5 the per-class / per-subject summary must reflect
    the P6 single-checkpoint reject rule's coverage, accepted accuracy and
    wrong-action rate — never a fabricated ``coverage=1.0`` shortcut. We
    :func:`apply_rule` with the loaded threshold, then group the resulting
    ``accepted`` / ``wrong_action`` columns by ``y_true`` or ``subject_id``.
    """
    if stitched.empty:
        return []
    scored = apply_rule(
        add_uncertainty_columns(stitched.copy()),
        RejectRule(float(rule.threshold)),
    )
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(by, sort=True):
        accepted = group["accepted"]
        accepted_n = int(accepted.sum())
        wrong_n = int(group["wrong_action"].sum())
        n = int(len(group))
        rows.append(
            {
                by: str(key),
                "n": n,
                "wrong_action_n": wrong_n,
                "wrong_action_rate": float(wrong_n) / n if n else 0.0,
                "accuracy": float(group["correct"].sum()) / n if n else 0.0,
                "accepted_n": accepted_n,
                "coverage": float(accepted_n) / n if n else 0.0,
                "accepted_accuracy": (
                    float(group.loc[accepted, "correct"].mean()) if accepted_n else None
                ),
                "accepted_error_rate": (
                    float(wrong_n) / accepted_n if accepted_n else None
                ),
                "p6_threshold": float(rule.threshold),
            }
        )
    return rows


def _worst_subjects(
    stitched: pd.DataFrame, *, rule: P6SingleRule
) -> dict[str, Any]:
    """Return the worst subject by FOUR distinct criteria simultaneously.

    Per Round-4 review the report must surface the truly worst subject
    by each of these metrics, not collapse them into a single dict:

    - ``by_wrong_action_rate`` — highest wrong_action_rate DESC (most wrong).
    - ``by_coverage``           — lowest coverage ASC (most rejected).
    - ``by_accepted_accuracy``  — lowest accepted_accuracy ASC (least accurate
      among the accepted subset).
    - ``by_raw_accuracy``       — lowest accuracy ASC (least accurate overall,
      irrespective of the rejection rule).

    Ties on each criterion are broken deterministically by ``subject_id``
    ASC so reruns produce the same ordering. ``None`` metrics (e.g. an
    ``accepted_accuracy`` for a subject with zero accepted records) are
    treated as ``0.0`` for sorting purposes; subjects with no accepted
    records are correctly considered worst-by-accepted_accuracy.
    """
    breakdown = _per_subject_breakdown(stitched, rule=rule)
    empty: dict[str, Any] = {
        "by_wrong_action_rate": None,
        "by_coverage": None,
        "by_accepted_accuracy": None,
        "by_raw_accuracy": None,
    }
    if not breakdown:
        return empty

    def _pick(key: str, *, descending: bool) -> dict[str, Any]:
        # ``or 0.0`` keeps None metrics in the sort (treated as worst-by
        # accuracy/coverage, but not as best-by-war — the descending flag
        # handles that direction correctly).
        metric_key = lambda row: float(row.get(key) or 0.0)  # noqa: E731
        return sorted(
            breakdown,
            key=lambda row: (
                -metric_key(row) if descending else metric_key(row),
                str(row["subject_id"]),
            ),
        )[0]

    return {
        "by_wrong_action_rate": _pick("wrong_action_rate", descending=True),
        "by_coverage": _pick("coverage", descending=False),
        "by_accepted_accuracy": _pick("accepted_accuracy", descending=False),
        "by_raw_accuracy": _pick("accuracy", descending=False),
    }


def _error_cases(stitched: pd.DataFrame) -> list[dict[str, Any]]:
    """Return every wrong record (with uncertainty diagnostics) sorted by confidence.

    Per Reviewer point #4 the runner must not pass an illegal threshold
    like ``1e9`` to :func:`error_cases`. ``threshold=0.0`` is a legal
    threshold (the lower bound of ``apply_rule``) that accepts every
    record; the subsequent ``~correct`` filter keeps only the wrong
    predictions, which is exactly what the report needs.
    """
    if stitched.empty:
        return []
    frame = error_cases(
        add_uncertainty_columns(stitched.copy()),
        threshold=0.0,
        high_confidence=0.90,
    )
    return frame.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Condition + seed summary blocks
# ---------------------------------------------------------------------------


def _condition_seed_summary(
    condition: P7Condition,
    seed: int,
    stitched: pd.DataFrame,
    clean_stitched: pd.DataFrame,
    *,
    p6_single_rule: P6SingleRule,
    p6_1_ensemble_rule: P61EnsembleRule,
) -> dict[str, Any]:
    """Compute the per-(condition, seed) stitched summary block."""
    record_metrics = _stitched_classification_metrics(stitched)
    p6_single = _stitched_p6_single(stitched, rule=p6_single_rule)
    p6_1 = _stitched_p6_1_ensemble(stitched, rule=p6_1_ensemble_rule)
    per_class = _per_class_breakdown(stitched, rule=p6_single_rule)
    per_subject = _per_subject_breakdown(stitched, rule=p6_single_rule)
    worst_subjects = _worst_subjects(stitched, rule=p6_single_rule)
    errors = _error_cases(stitched)

    clean_metrics = _stitched_classification_metrics(clean_stitched)
    delta = {
        "record_macro_f1": float(record_metrics["macro_f1"]) - float(clean_metrics["macro_f1"]),
        "record_balanced_accuracy": float(record_metrics["balanced_accuracy"])
        - float(clean_metrics["balanced_accuracy"]),
        "record_accuracy": float(record_metrics["accuracy"]) - float(clean_metrics["accuracy"]),
        "p6_single_wrong_action_rate": (
            float(p6_single["wrong_action_rate"]) - float(_stitched_p6_single(clean_stitched, rule=p6_single_rule)["wrong_action_rate"])
            if p6_single["wrong_action_rate"] is not None else None
        ),
    }
    return {
        "condition": condition.as_dict(),
        "seed": int(seed),
        "n_records": int(len(stitched)),
        "n_unique_records": int(stitched["record_id"].nunique()),
        "record_metrics": record_metrics,
        "delta_vs_clean": delta,
        "p6_single_rule": p6_single,
        "p6_1_ensemble_rule": p6_1,
        "per_class": per_class,
        "per_subject": per_subject,
        "worst_subjects": worst_subjects,
        "error_cases": errors,
    }


def _condition_seed_drift_stats(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate delta-vs-clean across seeds for one condition."""
    macros = np.asarray(
        [float(s["delta_vs_clean"]["record_macro_f1"]) for s in seed_summaries],
        dtype=float,
    )
    balanced = np.asarray(
        [float(s["delta_vs_clean"]["record_balanced_accuracy"]) for s in seed_summaries],
        dtype=float,
    )
    accuracy = np.asarray(
        [float(s["delta_vs_clean"]["record_accuracy"]) for s in seed_summaries],
        dtype=float,
    )
    return {
        "delta_macro_f1_mean": float(macros.mean()),
        "delta_macro_f1_std": float(macros.std()),
        "delta_macro_f1_worst": float(macros.min()),
        "delta_balanced_accuracy_mean": float(balanced.mean()),
        "delta_balanced_accuracy_std": float(balanced.std()),
        "delta_balanced_accuracy_worst": float(balanced.min()),
        "delta_accuracy_mean": float(accuracy.mean()),
        "delta_accuracy_std": float(accuracy.std()),
        "delta_accuracy_worst": float(accuracy.min()),
        "n_seeds": int(len(seed_summaries)),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _load_p6_rules_from_config(config: Mapping[str, Any]) -> tuple[P6SingleRule, P61EnsembleRule]:
    evidence_block = config["p6_evidence"]
    single = load_p6_single_rule(evidence_block["single_threshold_source"])
    ensemble = load_p6_1_ensemble_rule(evidence_block["ensemble_rule_source"])
    return single, ensemble


def run_popu_p7_robustness(
    parameters: Mapping[str, Any],
    seed: int,
    experiment_dir: Path,
) -> dict[str, Any]:
    """Execute the frozen P7 robustness evaluation.

    ``parameters`` is expected to be ``configs/analysis/popu_p7_robustness_v0.1.json``.
    The runner is deterministic from the frozen conditions + per-fold checkpoint
    seeds; the supplied ``seed`` is recorded in the manifest only.
    """
    config = dict(parameters)
    _validate_p7_config(config)
    seeds = [int(item) for item in config["seeds"]]
    conditions = parse_p7_conditions(config)
    p6_single_rule, p6_1_ensemble_rule = _load_p6_rules_from_config(config)

    evidence_root = resolve_evidence_root(
        config.get("evidence_root") or os.environ.get("P7_EVIDENCE_ROOT")
    )
    smoke_mode = bool(config.get("smoke_max_records"))
    # CLI narrowing lives under dedicated keys so the frozen repeats /
    # local_folds in the config block keep the FULL_REPEATS / FULL_LOCAL_FOLDS
    # invariant for downstream contract validation.
    narrowed_repeats = config.get("__narrowed_repeats")
    narrowed_local_folds = config.get("__narrowed_local_folds")
    has_narrowing = narrowed_repeats is not None or narrowed_local_folds is not None
    require_full = not smoke_mode and not has_narrowing
    # If the caller explicitly narrowed repeats/folds, honour that; otherwise
    # require all 15 folds.
    repeats_arg: list[int] | None
    folds_arg: list[int] | None
    if narrowed_repeats is not None:
        repeats_arg = [int(item) for item in narrowed_repeats]
    else:
        repeats_arg = list(FULL_REPEATS)
    if narrowed_local_folds is not None:
        folds_arg = [int(item) for item in narrowed_local_folds]
    else:
        folds_arg = list(FULL_LOCAL_FOLDS)
    if smoke_mode:
        require_full = False

    folds = resolve_fold_checkpoints(
        evidence_root,
        model_family=MODEL_FAMILY,
        repeats=repeats_arg,
        local_folds=folds_arg,
        require_full=require_full,
    )

    cohort_params = config.get("full_cohort_parameters")
    if not isinstance(cohort_params, Mapping):
        raise ValueError(
            "P7 parameters.full_cohort_parameters is required (path config + "
            "quality_manifest pinning mirroring the P5.2-C Full runner)."
        )
    cohort = load_full_cohort(cohort_params)
    device = resolve_device(str(config.get("device", "auto")))

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "folds").mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        experiment_dir / "config_used.json",
        {
            "p7_config": config,
            "evidence_root": str(evidence_root),
            "evidence_exp_id": EVIDENCE_EXP_ID,
            "p6_single_rule": p6_single_rule.as_dict(),
            "p6_1_ensemble_rule": p6_1_ensemble_rule.as_dict(),
        },
    )
    atomic_write_json(
        experiment_dir / "scope.json",
        {
            "repeats": [fold.repeat for fold in folds],
            "local_folds": [fold.local_fold for fold in folds],
            "seeds": seeds,
            "conditions": [condition.as_dict() for condition in conditions],
            "device": str(device),
            "frozen_protocol": PROTOCOL_NAME,
            "smoke_max_records": config.get("smoke_max_records"),
        },
    )

    clean_stitched_frames: list[pd.DataFrame] = []
    condition_seed_stitched_frames: dict[tuple[str, int], list[pd.DataFrame]] = defaultdict(list)
    fold_clean_summaries: list[dict[str, Any]] = []

    for fold_checkpoint in folds:
        fold_dir = experiment_dir / "folds" / f"repeat_{fold_checkpoint.repeat}" / f"fold_{fold_checkpoint.local_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model, normalizer = _load_fold_payload(fold_checkpoint, device)
        matrices, labels, sample_ids, record_ids, subject_ids = _load_outer_test_samples(
            cohort, outer_test_subjects=fold_checkpoint.outer_test_subjects
        )
        records = _group_records_by_record(sample_ids, record_ids, subject_ids, matrices, labels)
        if config.get("smoke_max_records") is not None:
            max_records = int(config["smoke_max_records"])
            if max_records > 0 and len(records) > max_records:
                records = records[:max_records]

        # CLEAN inference + cross-check.
        clean_snapshot_rows = _infer_records(
            model, normalizer, records, fold_checkpoint=fold_checkpoint, device=device
        )
        clean_record_rows = aggregate_record_rows(clean_snapshot_rows)
        clean_crosscheck = _assert_clean_matches_oof(
            clean_snapshot_rows,
            fold_checkpoint=fold_checkpoint,
            exhaustive=not smoke_mode,
        )
        clean_record_metrics = _record_metric_blocks(clean_record_rows)

        _write_csv_atomic(
            fold_dir / "clean" / "snapshot_predictions.csv",
            SNAPSHOT_COLUMNS,
            clean_snapshot_rows,
        )
        _write_csv_atomic(
            fold_dir / "clean" / "record_predictions.csv",
            RECORD_COLUMNS,
            clean_record_rows,
        )
        atomic_write_json(
            fold_dir / "clean" / "summary.json",
            {
                "repeat": fold_checkpoint.repeat,
                "local_fold": fold_checkpoint.local_fold,
                "model": MODEL_FAMILY,
                "n_snapshots": len(clean_snapshot_rows),
                "n_records": len(clean_record_rows),
                "record_metrics": clean_record_metrics,
                "oof_crosscheck": clean_crosscheck,
                "checkpoint_sha256": fold_checkpoint.expected_sha256,
                "checkpoint_size_bytes": fold_checkpoint.expected_size_bytes,
                "split_manifest_sha256": fold_checkpoint.split_manifest_sha256,
                "complete_sha256": fold_checkpoint.complete_sha256,
            },
        )

        clean_stitched_frames.append(pd.DataFrame(clean_record_rows))
        fold_clean_summaries.append(
            {
                "repeat": fold_checkpoint.repeat,
                "local_fold": fold_checkpoint.local_fold,
                "checkpoint_sha256": fold_checkpoint.expected_sha256,
                "complete_sha256": fold_checkpoint.complete_sha256,
                "split_manifest_sha256": fold_checkpoint.split_manifest_sha256,
                "n_clean_records": len(clean_record_rows),
                "clean_record_metrics": clean_record_metrics,
                "oof_crosscheck": clean_crosscheck,
            }
        )

        fold_condition_results: list[dict[str, Any]] = []
        for condition in conditions:
            for perturb_seed in seeds:
                perturbed_records = perturb_records(records, condition, perturb_seed)
                snapshot_rows = _infer_records(
                    model,
                    normalizer,
                    perturbed_records,
                    fold_checkpoint=fold_checkpoint,
                    device=device,
                )
                record_rows = aggregate_record_rows(snapshot_rows)
                p6_single = _record_p6_single_metrics(record_rows, rule=p6_single_rule)
                p6_1 = _record_p6_1_ensemble_metrics(record_rows, rule=p6_1_ensemble_rule)
                condition_seed_stitched_frames[
                    (condition.name, perturb_seed)
                ].append(pd.DataFrame(record_rows))

                cond_dir = fold_dir / condition.name / f"seed_{perturb_seed}"
                cond_dir.mkdir(parents=True, exist_ok=True)
                _write_csv_atomic(
                    cond_dir / "snapshot_predictions.csv",
                    SNAPSHOT_COLUMNS,
                    snapshot_rows,
                )
                _write_csv_atomic(
                    cond_dir / "record_predictions.csv",
                    RECORD_COLUMNS,
                    record_rows,
                )
                fold_condition_results.append(
                    {
                        "condition": condition.as_dict(),
                        "seed": perturb_seed,
                        "n_snapshots": len(snapshot_rows),
                        "n_records": len(record_rows),
                        "p6_single_rule": p6_single,
                        "p6_1_ensemble_rule": p6_1,
                    }
                )

        atomic_write_json(
            fold_dir / "summary.json",
            {
                "repeat": fold_checkpoint.repeat,
                "local_fold": fold_checkpoint.local_fold,
                "model": MODEL_FAMILY,
                "checkpoint_sha256": fold_checkpoint.expected_sha256,
                "complete_sha256": fold_checkpoint.complete_sha256,
                "split_manifest_sha256": fold_checkpoint.split_manifest_sha256,
                "n_clean_records": len(clean_record_rows),
                "clean_record_metrics": clean_record_metrics,
                "oof_crosscheck": clean_crosscheck,
                "p6_single_rule": p6_single_rule.as_dict(),
                "p6_1_ensemble_rule": p6_1_ensemble_rule.as_dict(),
                "condition_results": fold_condition_results,
            },
        )

    # -------------------------------------------------------------------
    # Stitch clean OOF across all 15 folds.
    # -------------------------------------------------------------------
    clean_stitched = _stitch_full_oof(clean_stitched_frames)
    clean_stitched_metrics = _stitched_classification_metrics(clean_stitched)

    # -------------------------------------------------------------------
    # Stitch per (condition, seed) and emit stitched summaries.
    # -------------------------------------------------------------------
    condition_seed_summaries: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for (condition_name, perturb_seed), frames in sorted(condition_seed_stitched_frames.items()):
        stitched = _stitch_full_oof(frames)
        condition = next(
            (item for item in conditions if item.name == condition_name), None
        )
        if condition is None:
            raise ValueError(f"Unknown condition {condition_name!r} in stitched frames.")
        summary = _condition_seed_summary(
            condition, perturb_seed, stitched, clean_stitched,
            p6_single_rule=p6_single_rule,
            p6_1_ensemble_rule=p6_1_ensemble_rule,
        )
        condition_seed_summaries.append(summary)

    # Group summaries by condition for the condition-level report.
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in condition_seed_summaries:
        by_condition[str(summary["condition"]["name"])].append(summary)
    for condition_name in sorted(by_condition):
        group = by_condition[condition_name]
        drift = _condition_seed_drift_stats(group)
        condition_summaries.append(
            {
                "condition_name": condition_name,
                "n_runs": int(len(group)),
                "record_metrics_stitched_means": {
                    key: float(np.asarray(
                        [float(s["record_metrics"][key]) for s in group], dtype=float
                    ).mean())
                    for key in ("accuracy", "balanced_accuracy", "macro_f1")
                },
                "delta_vs_clean": drift,
                "p6_single_rule_means": {
                    key: float(np.asarray(
                        [
                            float(s["p6_single_rule"][key])
                            for s in group
                            if s["p6_single_rule"][key] is not None
                        ], dtype=float,
                    ).mean())
                    for key in ("coverage", "accepted_accuracy", "wrong_action_rate")
                },
                "p6_1_ensemble_rule_means": {
                    key: float(np.asarray(
                        [
                            float(s["p6_1_ensemble_rule"][key])
                            for s in group
                            if s["p6_1_ensemble_rule"][key] is not None
                        ], dtype=float,
                    ).mean())
                    for key in ("coverage", "accepted_accuracy", "wrong_action_rate")
                },
                "seed_summaries": group,
            }
        )

    atomic_write_json(
        experiment_dir / "condition_comparison.json",
        {
            "model_family": MODEL_FAMILY,
            "frozen_protocol": PROTOCOL_NAME,
            "split_manifest_actual_sha256": folds[0].split_manifest_actual_sha256,
            "n_folds_resolved": len(folds),
            "n_total_folds_required": FULL_TOTAL_FOLDS,
            "stitching_policy": "pool_first_then_metric",
            "p6_single_rule": p6_single_rule.as_dict(),
            "p6_1_ensemble_rule": p6_1_ensemble_rule.as_dict(),
            "clean_stitched_metrics": clean_stitched_metrics,
            "clean_n_records_total": int(len(clean_stitched)),
            "fold_clean_summaries": fold_clean_summaries,
            "condition_summaries": condition_summaries,
            "decision": (
                "Sensitivity characterization only; software perturbations on "
                "PoPu 64x27 Tactilus matrices are not validation of a real "
                "low-density or faulty product sensor."
            ),
        },
    )

    return {
        "model_family": MODEL_FAMILY,
        "evidence_root": str(evidence_root),
        "n_folds_resolved": len(folds),
        "n_conditions": len(conditions),
        "n_seeds": len(seeds),
        "frozen_protocol": PROTOCOL_NAME,
        "p6_single_threshold": float(p6_single_rule.threshold),
        "p6_1_ensemble_threshold": float(p6_1_ensemble_rule.threshold),
    }


# ---------------------------------------------------------------------------
# Clean-only full-fold CPU reproduction (Reviewer point 7)
# ---------------------------------------------------------------------------


def run_clean_only_full_fold(
    parameters: Mapping[str, Any],
    *,
    experiment_dir: Path,
    repeat: int,
    local_fold: int,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Re-infer every outer-test record of one fold on CPU and compare to OOF.

    This is the clean-only reproducibility anchor that demonstrates the
    frozen P5.2-C evidence pack can be reloaded without GPU and produce the
    same record-level predictions and probabilities as the captured OOF.
    """
    config = dict(parameters)
    _validate_p7_config(config)
    p6_single_rule, _p6_1_ensemble_rule = _load_p6_rules_from_config(config)

    evidence_root = resolve_evidence_root(
        config.get("evidence_root") or os.environ.get("P7_EVIDENCE_ROOT")
    )
    folds = resolve_fold_checkpoints(
        evidence_root,
        model_family=MODEL_FAMILY,
        repeats=[int(repeat)],
        local_folds=[int(local_fold)],
        require_full=False,
    )
    if len(folds) != 1:
        raise ValueError(
            f"clean-only full-fold expects exactly one fold; resolved {len(folds)}."
        )
    fold_checkpoint = folds[0]

    cohort_params = config.get("full_cohort_parameters")
    if not isinstance(cohort_params, Mapping):
        raise ValueError("P7 parameters.full_cohort_parameters is required.")
    cohort = load_full_cohort(cohort_params)
    device = resolve_device("cpu")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = experiment_dir / "folds" / f"repeat_{fold_checkpoint.repeat}" / f"fold_{fold_checkpoint.local_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer = _load_fold_payload(fold_checkpoint, device)
    matrices, labels, sample_ids, record_ids, subject_ids = _load_outer_test_samples(
        cohort, outer_test_subjects=fold_checkpoint.outer_test_subjects
    )
    records = _group_records_by_record(sample_ids, record_ids, subject_ids, matrices, labels)

    snapshot_rows = _infer_records(
        model, normalizer, records, fold_checkpoint=fold_checkpoint, device=device
    )
    record_rows = aggregate_record_rows(snapshot_rows)
    crosscheck = _assert_clean_matches_oof(
        snapshot_rows,
        fold_checkpoint=fold_checkpoint,
        exhaustive=True,
    )
    record_metrics = _record_metric_blocks(record_rows)

    _write_csv_atomic(
        fold_dir / "clean" / "snapshot_predictions.csv",
        SNAPSHOT_COLUMNS,
        snapshot_rows,
    )
    _write_csv_atomic(
        fold_dir / "clean" / "record_predictions.csv",
        RECORD_COLUMNS,
        record_rows,
    )
    atomic_write_json(
        fold_dir / "clean" / "summary.json",
        {
            "repeat": fold_checkpoint.repeat,
            "local_fold": fold_checkpoint.local_fold,
            "model": MODEL_FAMILY,
            "n_snapshots": len(snapshot_rows),
            "n_records": len(record_rows),
            "record_metrics": record_metrics,
            "oof_crosscheck": crosscheck,
            "checkpoint_sha256": fold_checkpoint.expected_sha256,
            "checkpoint_size_bytes": fold_checkpoint.expected_size_bytes,
            "split_manifest_sha256": fold_checkpoint.split_manifest_sha256,
            "complete_sha256": fold_checkpoint.complete_sha256,
        },
    )
    atomic_write_json(
        experiment_dir / "summary.json",
        {
            "model_family": MODEL_FAMILY,
            "frozen_protocol": PROTOCOL_NAME,
            "repeat": fold_checkpoint.repeat,
            "local_fold": fold_checkpoint.local_fold,
            "n_records": len(record_rows),
            "n_snapshots": len(snapshot_rows),
            "record_metrics": record_metrics,
            "oof_crosscheck": crosscheck,
            "p6_single_rule": p6_single_rule.as_dict(),
            "checkpoint_sha256": fold_checkpoint.expected_sha256,
            "complete_sha256": fold_checkpoint.complete_sha256,
            "split_manifest_sha256": fold_checkpoint.split_manifest_sha256,
        },
    )

    return {
        "repeat": fold_checkpoint.repeat,
        "local_fold": fold_checkpoint.local_fold,
        "n_records": len(record_rows),
        "record_metrics": record_metrics,
        "oof_crosscheck": crosscheck,
    }


def _write_csv_atomic(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)