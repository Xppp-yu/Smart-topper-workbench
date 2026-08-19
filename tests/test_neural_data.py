"""Contract tests for the PoPu neural data module (P5.2-A1).

These tests use small synthetic records (not the full PoPu dataset) to verify the
data contract: frozen label order, matrix shape/dtype, metadata completeness,
subject-isolated splitting, train-only normalization, and horizontal-flip
augmentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topper_perception.neural.data import (
    FROZEN_LABELS,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    MatrixNormalizer,
    build_labeled_samples,
    flip_labels,
    horizontal_flip,
    subject_split,
    to_model_input,
    validate_subject_split,
)

ROWS, COLS = 64, 27


def _write_record(
    data_root: Path,
    subject_id: str,
    filename: str,
    position: str | None,
    *,
    n_snapshots: int = 2,
    seed: int = 0,
) -> Path:
    """Write one synthetic Tactilus JSON record under ``data_root/tactilus_data``."""
    subject_dir = data_root / "tactilus_data" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    snapshots = {}
    for i in range(n_snapshots):
        readings = rng.uniform(0.0, 1.0, size=ROWS * COLS).astype(float).tolist()
        snapshots[str(i)] = {"id": f"snap_{i}", "tactilus_readings": readings}
    record = {
        "tactilus_rows": ROWS,
        "tactilus_columns": COLS,
        "volunteer_id": subject_id,
        "variation": "v1",
        "snapshots": snapshots,
    }
    if position is not None:
        record["position"] = position
    path = subject_dir / filename
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_frozen_label_order() -> None:
    assert FROZEN_LABELS == ("empty", "supine", "prone", "left", "right")
    assert NUM_CLASSES == 5
    assert LABEL_TO_INDEX["left"] == 3
    assert LABEL_TO_INDEX["right"] == 4


def test_to_model_input_shape_and_dtype(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "subj_01", "rec.json", "supine", n_snapshots=3)
    samples = build_labeled_samples([path], tactilus_root=tmp_path)
    x, y = to_model_input(samples)

    assert x.shape == (3, 1, ROWS, COLS)
    assert x.dtype == np.float32
    assert y.shape == (3,)
    assert y.dtype == np.int64
    assert y.tolist() == [LABEL_TO_INDEX["supine"]] * 3


def test_build_labeled_samples_metadata_complete(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "subj_01", "rec_supine.json", "supine")
    samples = build_labeled_samples([path], tactilus_root=tmp_path)

    assert len(samples) == 2
    sample = samples[0]
    assert sample.subject_id == "subj_01"
    assert sample.posture == "supine"
    assert sample.record_id == "subj_01/rec_supine.json"
    assert sample.snapshot_index in (0, 1)
    assert sample.sample_id == f"popu-tactilus::subj_01/rec_supine.json#frame={sample.snapshot_index}"
    assert sample.matrix.shape == (ROWS, COLS)
    assert sample.source_file.endswith("rec_supine.json")


def test_unlabeled_record_is_excluded(tmp_path: Path) -> None:
    labeled = _write_record(tmp_path, "subj_a", "labeled.json", "left")
    others = _write_record(tmp_path, "subj_b", "others.json", None)  # no position
    samples = build_labeled_samples([labeled, others], tactilus_root=tmp_path)

    assert len(samples) == 2
    assert all(s.subject_id == "subj_a" for s in samples)
    assert all(s.posture == "left" for s in samples)


def test_build_labeled_samples_deterministic(tmp_path: Path) -> None:
    a = _write_record(tmp_path, "subj_02", "a.json", "prone")
    b = _write_record(tmp_path, "subj_01", "b.json", "right")

    first = [s.sample_id for s in build_labeled_samples([a, b], tactilus_root=tmp_path)]
    second = [s.sample_id for s in build_labeled_samples([b, a], tactilus_root=tmp_path)]

    assert first == second


def test_subject_split_no_overlap() -> None:
    subject_ids = ["s1"] * 3 + ["s2"] * 2 + ["s3"] * 4 + ["s4"] * 1 + ["s5"] * 2
    split = subject_split(subject_ids, val_ratio=0.2, test_ratio=0.2, seed=42)
    validate_subject_split(split, subject_ids)  # must not raise

    train = {subject_ids[i] for i in split.train_indices}
    val = {subject_ids[i] for i in split.val_indices}
    test = {subject_ids[i] for i in split.test_indices}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    # Every subject's samples land entirely within one split.
    for subject in {"s1", "s2", "s3", "s4", "s5"}:
        positions = {i for i, sid in enumerate(subject_ids) if sid == subject}
        assert (
            positions <= set(split.train_indices.tolist())
            or positions <= set(split.val_indices.tolist())
            or positions <= set(split.test_indices.tolist())
        )


def test_normalizer_fits_train_only() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(10.0, 3.0, size=(100, ROWS, COLS)).astype(np.float32)
    val = rng.normal(50.0, 5.0, size=(20, ROWS, COLS)).astype(np.float32)

    normalizer = MatrixNormalizer()
    normalizer.fit(train)

    # Statistics are derived from train, nothing else.
    assert normalizer.mean_ == pytest.approx(float(np.mean(train)))
    assert normalizer.std_ == pytest.approx(float(np.std(train)))

    out_train = normalizer.transform(train)
    assert float(np.mean(out_train)) == pytest.approx(0.0, abs=1e-3)
    assert float(np.std(out_train)) == pytest.approx(1.0, abs=1e-2)

    # Validation is standardized with train's frozen stats, not its own.
    out_val = normalizer.transform(val)
    assert np.allclose(out_val, (val - normalizer.mean_) / normalizer.std_)


def test_horizontal_flip_swaps_left_and_right() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 1, ROWS, COLS)).astype(np.float32)
    labels = np.array([3, 4, 0, 2])  # left, right, empty, prone

    flipped, flipped_labels = horizontal_flip(x, labels)

    assert flipped.shape == x.shape
    assert np.array_equal(flipped[..., 0], x[..., -1])
    assert np.array_equal(flipped[..., -1], x[..., 0])
    assert flipped_labels.tolist() == [4, 3, 0, 2]


def test_flip_labels_leaves_non_side_postures_unchanged() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    assert flip_labels(labels).tolist() == [0, 1, 2, 4, 3]

    with pytest.raises(ValueError):
        flip_labels(np.array([5]))
