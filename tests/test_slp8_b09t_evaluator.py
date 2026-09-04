from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from topper_perception.evaluation.slp8_b09t_evaluator import (
    B09TEvaluatorError,
    evaluate_hard_predictions,
    hard_plurality_vote,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_slp8_b09t_evaluator.py"


def test_vote_majority_all_different_and_unanimous() -> None:
    p42 = np.zeros((1, 192, 84), dtype=np.int64)
    p123 = p42.copy(); p2026 = p42.copy()
    p42[0, 0, :3] = [1, 2, 3]
    p123[0, 0, :3] = [1, 4, 3]
    p2026[0, 0, :3] = [5, 6, 3]
    primary, secondary, three_way = hard_plurality_vote({42: p42, 123: p123, 2026: p2026})
    assert primary[0, 0, :3].tolist() == [1, 2, 3]
    assert secondary[0, 0, :3].tolist() == [-1, -1, 3]
    assert three_way[0, 0, :3].tolist() == [False, True, False]


@pytest.mark.parametrize("bad", [
    {123: np.zeros((1, 192, 84), dtype=np.int64), 42: np.zeros((1, 192, 84), dtype=np.int64), 2026: np.zeros((1, 192, 84), dtype=np.int64)},
    {42: np.zeros((1, 192, 84)), 123: np.zeros((1, 192, 84)), 2026: np.zeros((1, 192, 84))},
    {42: np.full((1, 192, 84), 9), 123: np.zeros((1, 192, 84), dtype=np.int64), 2026: np.zeros((1, 192, 84), dtype=np.int64)},
    {42: np.zeros((1, 191, 84), dtype=np.int64), 123: np.zeros((1, 191, 84), dtype=np.int64), 2026: np.zeros((1, 191, 84), dtype=np.int64)},
])
def test_vote_rejects_contract_drift(bad) -> None:
    with pytest.raises(B09TEvaluatorError):
        hard_plurality_vote(bad)


def test_evaluator_metrics_and_tie_audit() -> None:
    labels = np.zeros((2, 192, 84), dtype=np.int64)
    labels[0, :96] = 1
    labels[1, 96:] = 2
    p42 = labels.copy(); p123 = labels.copy(); p2026 = labels.copy()
    p123[0, 0, 0] = 3; p2026[0, 0, 0] = 4
    out = evaluate_hard_predictions(labels, {42: p42, 123: p123, 2026: p2026}, ["A", "B"])
    assert out["sample_count"] == 2
    assert out["primary"]["three_way_disagreement_pixel_count"] == 1
    assert out["primary"]["three_way_disagreement_pixel_fraction"] == pytest.approx(1 / labels.size)
    assert set(out["primary"]["per_subject_fixed_foreground_macro_iou"]) == {"A", "B"}
    assert [row["class_id"] for row in out["primary"]["per_region"]] == list(range(1, 9))
    assert "background_iou" in out["primary"]


def test_evaluator_rejects_non_frozen_label_shape() -> None:
    labels = np.zeros((1, 192, 83), dtype=np.int64)
    predictions = {seed: labels.copy() for seed in (42, 123, 2026)}
    with pytest.raises(B09TEvaluatorError, match="frozen shape"):
        evaluate_hard_predictions(labels, predictions, ["A"])


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True)


def test_validate_only_writes_nothing(tmp_path: Path) -> None:
    result = _run("--validate-only")
    assert result.returncode == 0
    assert "TEST=0 GPU_NOT_RUN EXECUTION_NOT_AUTHORIZED" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_synthetic_smoke_writes_only_synthetic_summary(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = _run("--synthetic-smoke", "--experiment-id", "SMOKE-B09T-R01", "--output-dir", str(output))
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "synthetic_summary.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic_no_test"
    assert payload["test_access"] is False and payload["test_rows"] == 0
    assert payload["gpu_run"] is False
    assert [path.name for path in output.iterdir()] == ["synthetic_summary.json"]


def test_runner_refuses_formal_id_existing_output_and_extra_args(tmp_path: Path) -> None:
    existing = tmp_path / "existing"; existing.mkdir()
    assert _run("--synthetic-smoke", "--experiment-id", "EXP-B09T-R01", "--output-dir", str(tmp_path / "x")).returncode == 2
    assert _run("--synthetic-smoke", "--experiment-id", "SMOKE-B09T-R01", "--output-dir", str(existing)).returncode == 2
    assert _run("--validate-only", "--experiment-id", "SMOKE-B09T-R01").returncode == 2


def test_runner_has_no_real_test_or_gpu_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    evaluator = (ROOT / "src/topper_perception/evaluation/slp8_b09t_evaluator.py").read_text(encoding="utf-8")
    for forbidden in ("enable_test_access", "load_test=True", "load_b01_freeze_tables", "--run-authorized", "cuda"):
        assert forbidden not in source
        assert forbidden not in evaluator
