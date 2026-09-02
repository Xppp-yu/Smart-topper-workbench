from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_slp8_b10_consensus import analyze


def _write(path: Path, seed: int, pred, target=None, ids=None, candidate="winner"):
    target = np.asarray(target if target is not None else [[[1, 1], [0, 0]]])
    ids = np.asarray(ids if ids is not None else ["SLP:danaLab:00001:uncover:000001"], dtype=object)
    np.savez_compressed(path, predictions=np.asarray(pred), targets=target,
                        sample_ids=ids, candidate=candidate, seed=seed)


def test_consensus_metrics_are_hand_checkable(tmp_path: Path):
    target = [[[1, 1], [0, 0]]]
    preds = [
        [[[1, 1], [0, 1]]],
        [[[1, 2], [0, 1]]],
        [[[1, 1], [0, 1]]],
    ]
    paths = []
    for seed, pred in zip((42, 123, 2026), preds):
        path = tmp_path / f"{seed}.npz"; _write(path, seed, pred, target); paths.append(path)
    summary, subjects, errors = analyze(paths)
    assert summary["samples"] == 1
    assert summary["unanimous_pixels"]["coverage"] == pytest.approx(0.75)
    assert summary["all_pixels"]["error_rate"] == pytest.approx(0.25)
    assert summary["unanimous_error_fraction_of_raw_errors"] == pytest.approx(1.0)
    assert subjects[0]["subject_id"] == "00001"
    assert errors[0]["unanimous_wrong_pixels"] == 1


def test_target_drift_fails_closed(tmp_path: Path):
    paths = []
    for seed in (42, 123, 2026):
        path = tmp_path / f"{seed}.npz"
        target = [[[1, 1], [0, 0 if seed != 2026 else 1]]]
        _write(path, seed, [[[1, 1], [0, 0]]], target)
        paths.append(path)
    with pytest.raises(ValueError, match="target mismatch"):
        analyze(paths)
