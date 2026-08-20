"""Run P6 reject/uncertainty analysis from an extracted Full OOF evidence tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from topper_perception.neural.p6_reject import (
    choose_threshold,
    confusion_matrix,
    error_cases,
    grouped_metrics,
    load_record_oof,
    threshold_metrics_for,
    threshold_table,
    threshold_table_for,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, default=Path("configs/analysis/popu_p6_reject_v0.1.json"))
    args = parser.parse_args()
    config = json.loads(args.threshold_config.read_text(encoding="utf-8"))
    paths = sorted(args.oof_root.glob("outputs/experiments/*/folds/repeat_*/fold_*/small_resnet/record_predictions.csv"))
    if len(paths) != 15:
        raise ValueError(f"Expected 15 Small ResNet record OOF files, found {len(paths)}")
    frame = load_record_oof(paths)
    development = frame[frame["repeat"].isin(config["development_repeats"])].copy()
    evaluation = frame[frame["repeat"].isin(config["evaluation_repeats"])].copy()
    thresholds = config["threshold_grid"]
    constraints = config["selection_constraints"]
    rule, selected, rationale = choose_threshold(development, thresholds, **constraints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_table(frame, thresholds).to_csv(args.output_dir / "threshold_table_all_repeats.csv", index=False)
    threshold_table(development, thresholds).to_csv(args.output_dir / "threshold_table_development.csv", index=False)
    threshold_table(evaluation, thresholds).to_csv(args.output_dir / "threshold_table_evaluation.csv", index=False)
    alternative_specs = {
        "max_probability": (thresholds, "ge"),
        "top2_margin": ([0.00, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99], "ge"),
        "normalized_entropy": ([0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50], "le"),
    }
    alternative_rows = []
    for metric, (grid, direction) in alternative_specs.items():
        dev_table = threshold_table_for(development, metric, grid, accept_when=direction)
        feasible = dev_table[
            (dev_table["wrong_action_rate"] <= constraints["max_wrong_action_rate"])
            & (dev_table["accepted_accuracy"].fillna(0) >= constraints["min_accepted_accuracy"])
            & (dev_table["coverage"] >= constraints["min_coverage"])
        ]
        if feasible.empty:
            continue
        selected_alt = feasible.sort_values("threshold", ascending=(direction == "ge")).iloc[0]
        alt_threshold = float(selected_alt["threshold"])
        alternative_rows.append({
            "metric": metric, "threshold": alt_threshold,
            **{f"development_{k}": v for k, v in selected_alt.to_dict().items() if k not in {"metric", "threshold"}},
            **{f"evaluation_{k}": v for k, v in threshold_metrics_for(evaluation, metric, alt_threshold, accept_when=direction).items() if k not in {"metric", "threshold"}},
        })
    pd.DataFrame(alternative_rows).to_csv(args.output_dir / "alternative_rule_comparison.csv", index=False)
    grouped_metrics(frame, "y_true", rule.confidence_threshold).to_csv(args.output_dir / "per_class.csv", index=False)
    grouped_metrics(frame, "subject_id", rule.confidence_threshold).to_csv(args.output_dir / "per_subject.csv", index=False)
    grouped_metrics(evaluation, "y_true", rule.confidence_threshold).to_csv(args.output_dir / "per_class_evaluation.csv", index=False)
    grouped_metrics(evaluation, "subject_id", rule.confidence_threshold).to_csv(args.output_dir / "per_subject_evaluation.csv", index=False)
    confusion_matrix(frame, rule.confidence_threshold).to_csv(args.output_dir / "confusion_matrix_all_repeats.csv", index=False)
    confusion_matrix(evaluation, rule.confidence_threshold).to_csv(args.output_dir / "confusion_matrix_evaluation.csv", index=False)
    errors = error_cases(frame, rule.confidence_threshold, high_confidence=config["high_confidence_error_threshold"])
    errors.to_csv(args.output_dir / "error_cases.csv", index=False)
    errors[errors["high_confidence_error"]].to_csv(args.output_dir / "high_confidence_errors.csv", index=False)
    frame.to_csv(args.output_dir / "record_uncertainty_all_repeats.csv", index=False)
    subject_eval = grouped_metrics(evaluation, "subject_id", rule.confidence_threshold)
    class_eval = grouped_metrics(evaluation, "y_true", rule.confidence_threshold)
    gates = config.get("fairness_gates", {})
    fairness = {
        "min_subject_coverage": float(subject_eval["coverage"].min()),
        "min_subject_accepted_accuracy": float(subject_eval["accepted_accuracy"].min()),
        "max_subject_wrong_action_rate": float(subject_eval["wrong_action_rate"].max()),
        "class_coverage_range": float(class_eval["coverage"].max() - class_eval["coverage"].min()),
    }
    fairness["passed"] = (
        fairness["min_subject_coverage"] >= gates["evaluation_min_subject_coverage"]
        and fairness["min_subject_accepted_accuracy"] >= gates["evaluation_min_subject_accepted_accuracy"]
        and fairness["max_subject_wrong_action_rate"] <= gates["evaluation_max_subject_wrong_action_rate"]
        and fairness["class_coverage_range"] <= gates["evaluation_max_class_coverage_range"]
    )
    summary = {
        "model": config["model"], "level": config["level"], "n_records": len(frame),
        "n_records_per_repeat": {str(k): int(v) for k, v in frame.groupby("repeat").size().items()},
        "development_repeats": config["development_repeats"],
        "evaluation_repeats": config["evaluation_repeats"],
        "selected_rule": {"name": rule.name, "confidence_threshold": rule.confidence_threshold},
        "selection": selected, "selection_rationale": rationale,
        "evaluation": threshold_table(evaluation, [rule.confidence_threshold]).iloc[0].to_dict(),
        "all_repeats": threshold_table(frame, [rule.confidence_threshold]).iloc[0].to_dict(),
        "high_confidence_error_count": int(errors["high_confidence_error"].sum()),
        "error_count": int(len(errors)),
        "fairness_gates": gates,
        "fairness_evaluation": fairness,
        "p6_final_acceptance": bool(fairness["passed"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
