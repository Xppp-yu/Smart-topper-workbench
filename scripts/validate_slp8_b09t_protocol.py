"""Fail-closed validation for the B09T protocol. This script never loads TEST."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_SEEDS = {42: (15, "633aed4a25aa2cfc42208ef3c610a78aed3569acf0d75fddd47361623e655af3"),
                  123: (20, "e63415455816ea14dbbec4c54e9fd3c6c2f48be08de96fc8e60d2e1e94f7ffd5"),
                  2026: (12, "1ce88a9b1b4797bd158795f3e796e3682970ba881e76d7fc8759f70bb2c7578f")}


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(data.get("status") == "PROTOCOL_ONLY_TEST_NOT_AUTHORIZED", "status must keep TEST unauthorized")
    require(data.get("evaluation_mode") == "single_one_time_final_evaluation", "evaluation must be single and one-time")
    require(data.get("model_family") == "slp8_deeplabv3plus_lite_v0.1", "model family drift")
    require(HEX40.fullmatch(str(data.get("b11f_runner_git_commit", ""))) is not None, "runner SHA must be 40 lowercase hex")
    require(HEX64.fullmatch(str(data.get("b01_freeze_manifest_sha256", ""))) is not None, "B01 SHA must be 64 lowercase hex")
    cohort = data.get("test_cohort", {})
    require(cohort.get("expected_subjects") == 11 and cohort.get("expected_samples") == 495, "frozen TEST structural counts drift")
    require(cohort.get("subject_isolation_required") is True, "subject isolation must be required")

    checkpoints = data.get("checkpoints", [])
    observed = {item.get("seed"): (item.get("fixed_epochs"), item.get("sha256")) for item in checkpoints if isinstance(item, dict)}
    require(len(checkpoints) == 3 and observed == EXPECTED_SEEDS, "checkpoint seed/epoch/SHA set drift")
    require(all(HEX64.fullmatch(str(item.get("sha256", ""))) for item in checkpoints), "checkpoint SHA must be 64 lowercase hex")

    pred = data.get("prediction_contract", {})
    require(pred.get("primary") == "per_pixel_majority_vote_across_three_seed_hard_predictions", "primary prediction drift")
    require(pred.get("class_order") == list(range(9)), "class order must be 0..8")
    require("not_probability_not_OOD_not_safety" in str(pred.get("unknown_semantics", "")), "UNKNOWN limitations missing")

    metrics = data.get("frozen_metrics", {})
    require(metrics.get("primary") == "pooled_fixed_foreground_macro_iou", "primary metric drift")
    require(metrics.get("foreground_class_ids") == list(range(1, 9)), "foreground classes must be 1..8")
    require(metrics.get("zero_division") == 0, "zero division must be 0")
    require(metrics.get("empty_foreground_class_policy") == "include_as_zero", "empty classes must count as zero")

    gate = data.get("authorization_gate", {})
    require(gate.get("test_authorized") is False, "test_authorized must be strict false")
    require(gate.get("execution_authorized") is False, "execution_authorized must be strict false")
    require(gate.get("load_test") is False, "load_test must be strict false")
    require(gate.get("required_purpose_literal") == "final_evaluation", "purpose literal drift")
    require(len(gate.get("owner_must_freeze", [])) == 8, "Owner exact authorization fields incomplete")

    anti = data.get("anti_adaptation", {})
    require(all(anti.get(key) is True for key in (
        "no_parameter_or_threshold_tuning", "no_candidate_or_metric_change",
        "no_test_driven_rerun", "failure_requires_new_task_and_pollution_review")), "anti-adaptation gate incomplete")
    require("DONE.json_or_FAILED.json" in data.get("required_outputs", []), "terminal evidence output missing")
    require(len(data.get("prohibited_conclusions", [])) == 6, "prohibited conclusions incomplete")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    errors = validate(args.protocol)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("B09T_PROTOCOL_VALIDATION_FAILED TEST_DENIED EXECUTION_NOT_AUTHORIZED")
        return 1
    print("B09T_PROTOCOL_VALIDATION_PASSED TEST_DENIED EXECUTION_NOT_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
