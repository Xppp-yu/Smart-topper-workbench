"""Validate the frozen SLP8 B07 Full protocol without reading TEST data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATES = [
    "slp8_deeplabv3plus_lite_v0.1",
    "slp8_resunet_lite_v0.1",
]
EXPECTED_SEEDS = [42, 123, 2026]
EXPECTED_DEV_SUBJECTS = 91
EXPECTED_DEV_SAMPLES = 4095


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(protocol_path: Path) -> list[str]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fold_path = protocol_path.parents[2] / protocol["fold_contract"]["manifest_path"]
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(protocol.get("protocol") == "B07", "protocol must be B07")
    require(protocol.get("execution_authorized") is False, "execution must be unauthorized")
    require(protocol.get("runner_implemented") is False, "runner must be unimplemented")
    test = protocol.get("test_access", {})
    require(test.get("allowed") is False and test.get("load_test") is False, "TEST must be denied")
    require(all(test.get(k) == 0 for k in ("expected_rows", "expected_labels", "expected_onehot")), "TEST counts must be zero")

    require(_sha256(fold_path) == protocol["fold_contract"]["manifest_sha256"], "fold manifest SHA mismatch")
    require(folds.get("test_access") == "DENIED", "fold manifest TEST access must be DENIED")
    require(folds.get("development_subject_count") == EXPECTED_DEV_SUBJECTS, "development subject count mismatch")
    require(folds.get("development_sample_count") == EXPECTED_DEV_SAMPLES, "development sample count mismatch")
    fold_rows = folds.get("folds", [])
    require(len(fold_rows) == 5, "exactly five folds required")
    subject_ids = [sid for fold in fold_rows for sid in fold.get("val_subject_ids", [])]
    require(len(subject_ids) == EXPECTED_DEV_SUBJECTS, "fold subject total must be 91")
    require(len(set(subject_ids)) == EXPECTED_DEV_SUBJECTS, "fold subjects must be unique")
    require(sum(int(f["val_sample_count"]) for f in fold_rows) == EXPECTED_DEV_SAMPLES, "fold sample total must be 4095")
    require(all(int(f["val_sample_count"]) == int(f["val_subject_count"]) * 45 for f in fold_rows), "each fold must have 45 samples per subject")

    candidates = [c.get("name") for c in protocol.get("candidates", [])]
    require(candidates == EXPECTED_CANDIDATES, "candidate order/set mismatch")
    training = protocol.get("training_contract", {})
    require(training.get("seeds") == EXPECTED_SEEDS, "seeds mismatch")
    require(training.get("augmentation_policy") == "none", "augmentation must remain none")
    require(training.get("all_fold_seed_units_must_succeed") is True, "all units must succeed")
    matrix = protocol.get("execution_matrix", {})
    require(matrix.get("total_units") == 30, "execution matrix must contain 30 units")
    require(matrix.get("total_units") == matrix.get("candidates") * matrix.get("folds") * matrix.get("seeds"), "execution matrix multiplication mismatch")

    metrics = protocol.get("metrics", {})
    require("pooled" in str(metrics.get("primary", "")).lower(), "primary must use pooled OOF")
    require(metrics.get("fold_average_is_primary") is False, "fold average cannot be primary")
    require(metrics.get("failed_units_may_be_dropped") is False, "failed units cannot be dropped")
    selection = protocol.get("selection_rule", {})
    require(selection.get("output_candidate_count") == 1, "Full must freeze one candidate")
    require(math.isclose(float(selection.get("near_tie_if_absolute_difference_lt", -1)), 0.02), "near-tie margin must be 0.02")
    require(selection.get("no_test_in_selection") is True, "selection must not use TEST")

    budget = protocol.get("resource_budget", {})
    require(budget.get("max_wall_minutes_per_candidate") == 15 * 5 * 3, "candidate budget multiplication mismatch")
    require(budget.get("max_wall_minutes_total") == 15 * 5 * 3 * 2, "total budget multiplication mismatch")
    require(budget.get("max_peak_cuda_mb") == 8192, "CUDA budget mismatch")
    identity = protocol.get("identity_contract", {})
    required = set(identity.get("required_fields", []))
    require({"experiment_id", "git_commit", "config_sha256", "data_manifest_sha256", "split_sha256", "fold_manifest_sha256", "model_version", "candidate", "fold_id", "seed"} <= required, "identity fields incomplete")
    require(identity.get("git_dirty_must_be") is False, "git must be clean")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    errors = validate(args.protocol)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("B07_PROTOCOL_VALIDATION_PASSED TEST=0 folds=5 candidates=2 seeds=3 units=30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
