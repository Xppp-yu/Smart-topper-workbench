"""Run bounded P6.1 temperature calibration and ensemble consistency analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from topper_perception.neural.p6_1 import (
    aggregate_repeat_ensemble,
    calibrated_frame,
    deterministic_subject_split,
    per_subject_metrics,
    select_temperature,
    select_threshold,
    selective_metrics,
)
from topper_perception.neural.p6_reject import load_record_oof


def gate(per_subject: pd.DataFrame, gates: dict[str, float]) -> dict[str, object]:
    result = {
        "min_subject_coverage": float(per_subject["coverage"].min()),
        "min_subject_accepted_accuracy": float(per_subject["accepted_accuracy"].min()),
        "max_subject_wrong_action_rate": float(per_subject["wrong_action_rate"].max()),
    }
    result["passed"] = (
        result["min_subject_coverage"] >= gates["evaluation_min_subject_coverage"]
        and result["min_subject_accepted_accuracy"] >= gates["evaluation_min_subject_accepted_accuracy"]
        and result["max_subject_wrong_action_rate"] <= gates["evaluation_max_subject_wrong_action_rate"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/popu_p6_1_calibration_v0.1.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = sorted(args.oof_root.glob("outputs/experiments/*/folds/repeat_*/fold_*/small_resnet/record_predictions.csv"))
    oof = load_record_oof(paths)
    ensemble = aggregate_repeat_ensemble(oof)
    development_subjects, evaluation_subjects = deterministic_subject_split(
        ensemble["subject_id"].astype(str), seed=config["seed"],
        evaluation_count=config["evaluation_subject_count"],
    )
    development = ensemble[ensemble["subject_id"].isin(development_subjects)].copy()
    evaluation = ensemble[ensemble["subject_id"].isin(evaluation_subjects)].copy()
    temperature, temperature_table = select_temperature(development, config["temperature_grid"])
    development = calibrated_frame(development, temperature)
    evaluation = calibrated_frame(evaluation, temperature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temperature_table.to_csv(args.output_dir / "temperature_grid.csv", index=False)
    recurrence = ensemble.groupby("repeat_error_count", sort=True).size().rename("record_count").reset_index()
    recurrence.to_csv(args.output_dir / "cross_repeat_error_recurrence.csv", index=False)
    comparison = []
    fairness = {}
    for name, require_unanimous in (("calibrated_mean", False), ("calibrated_mean_plus_unanimous", True)):
        threshold, table = select_threshold(
            development, config["threshold_grid"], config["selection_constraints"],
            require_unanimous=require_unanimous,
        )
        table.to_csv(args.output_dir / f"threshold_table_{name}.csv", index=False)
        eval_metric = selective_metrics(evaluation, threshold, require_unanimous=require_unanimous)
        subject_table = per_subject_metrics(evaluation, threshold, require_unanimous=require_unanimous)
        subject_table.to_csv(args.output_dir / f"per_subject_{name}.csv", index=False)
        fairness[name] = gate(subject_table, config["fairness_gates"])
        comparison.append({"rule": name, "temperature": temperature, **eval_metric, **{f"gate_{k}": v for k, v in fairness[name].items()}})
    pd.DataFrame(comparison).to_csv(args.output_dir / "rule_comparison.csv", index=False)
    ensemble.to_csv(args.output_dir / "record_ensemble_diagnostics.csv", index=False)
    summary = {
        "temperature": temperature,
        "development_subjects": list(development_subjects),
        "evaluation_subjects": list(evaluation_subjects),
        "recurrence": recurrence.to_dict(orient="records"),
        "rules": comparison,
        "any_rule_passed_fairness": any(bool(value["passed"]) for value in fairness.values()),
        "scope": config["scope"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
