"""Unit tests for the B04A protocol contract validator (R03).

Each test is targeted at one of the R02+R03-detector branches. The tests use
small synthetic configs that intentionally violate one rule at a time, then
assert that the validator catches the violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_b04a_protocol import validate  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "experiments" / "slp8_pm_architecture_expansion_mini_v0.1.json"


def _load_good() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _run(cfg: dict) -> tuple[list[str], list[str]]:
    return validate(cfg)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_validator_passes_on_current_config() -> None:
    cfg = _load_good()
    errors, oks = _run(cfg)
    assert not errors, f"Expected PASS, got errors: {errors}"
    assert oks, "Expected at least one OK message"
    assert any("all required top-level fields present" in o for o in oks)


# ---------------------------------------------------------------------------
# R02.1 dtype inconsistency
# ---------------------------------------------------------------------------

def test_dtype_inconsistency_storage_eq_input_caught() -> None:
    cfg = _load_good()
    cfg["dataset"]["pressure_storage_dtype"] = "float32"
    cfg["dataset"]["model_input_dtype"] = "float32"
    errors, _ = _run(cfg)
    assert any("dtype mismatch" in e for e in errors), errors


def test_dtype_inconsistency_missing_conversion_policy_caught() -> None:
    cfg = _load_good()
    del cfg["dataset"]["dtype_conversion_policy"]
    errors, _ = _run(cfg)
    assert any("dtype_conversion_policy" in e for e in errors), errors


def test_dtype_conversion_rule_must_mention_both_sites() -> None:
    cfg = _load_good()
    cfg["dataset"]["dtype_conversion_policy"]["rule"] = (
        "convert from float64 to float32 somewhere in the loader"
    )
    errors, _ = _run(cfg)
    assert any("normalization" in e and "augmentation" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.2 candidate augmentation inconsistency
# ---------------------------------------------------------------------------

def test_candidate_augmentation_must_all_be_none() -> None:
    cfg = _load_good()
    cfg["candidates"][1]["augmentation_policy"] = "light"
    errors, _ = _run(cfg)
    assert any("augmentation_policy='none'" in e for e in errors), errors


def test_candidate_augmentation_blocks_must_be_identical() -> None:
    cfg = _load_good()
    cfg["candidates"][1]["augmentation"] = {
        "policy": "none",
        "description": "different description",
        "train": None,
        "val": None,
    }
    errors, _ = _run(cfg)
    assert any("Augmentation blocks differ" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.3 seed inconsistency
# ---------------------------------------------------------------------------

def test_seeds_must_be_42_123_2026() -> None:
    cfg = _load_good()
    cfg["training"]["seeds"] = [42]
    errors, _ = _run(cfg)
    assert any("seeds must be [42, 123, 2026]" in e for e in errors), errors


def test_augmentation_policy_per_candidate_must_be_none_for_all() -> None:
    cfg = _load_good()
    cfg["training"]["augmentation_policy_per_candidate"][
        "slp8_resunet_lite_v0.1"
    ] = "light"
    errors, _ = _run(cfg)
    assert any("slp8_resunet_lite_v0.1" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.4 missing 3_feasible
# ---------------------------------------------------------------------------

def test_missing_3_feasible_caught() -> None:
    cfg = _load_good()
    del cfg["feasibility_gate"]["decision_rules"]["3_feasible"]
    errors, _ = _run(cfg)
    assert any("3_feasible" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.5 failure handling
# ---------------------------------------------------------------------------

def test_all_seeds_must_succeed_must_be_true() -> None:
    cfg = _load_good()
    cfg["feasibility_gate"]["all_seeds_must_succeed"] = False
    errors, _ = _run(cfg)
    assert any("all_seeds_must_succeed must be true" in e for e in errors), errors


def test_candidate_feasibility_rule_must_mention_all_seeds() -> None:
    cfg = _load_good()
    cfg["candidate_decision_aggregation"]["candidate_feasibility_rule"] = (
        "candidate_feasible = majority of seeds pass"
    )
    errors, _ = _run(cfg)
    assert any(
        "all registered seeds" in e or "all_seeds" in e or "candidate_feasibility_rule" in e
        for e in errors
    ), errors


def test_macro_iou_mean_formula_must_mention_all_three() -> None:
    cfg = _load_good()
    cfg["candidate_decision_aggregation"]["macro_iou_mean_formula"] = (
        "mean over remaining successful seeds"
    )
    errors, _ = _run(cfg)
    assert errors, errors


# ---------------------------------------------------------------------------
# R02.6 resource budget
# ---------------------------------------------------------------------------

def test_resource_budget_multiplication_must_be_consistent() -> None:
    cfg = _load_good()
    cfg["resource_budget"]["total_wall_minutes"] = 999
    errors, _ = _run(cfg)
    assert any("inconsistent" in e for e in errors), errors


def test_resource_budget_must_have_candidate_and_total_accumulators() -> None:
    cfg = _load_good()
    del cfg["resource_budget"]["checkpoint_resume_keeps_accumulators"]["total_level"]
    errors, _ = _run(cfg)
    assert any("total_level" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.7 identity carrier format
# ---------------------------------------------------------------------------

def test_identity_carrier_format_must_cover_all_four_kinds() -> None:
    cfg = _load_good()
    del cfg["identity_hard_gate"]["carrier_format_by_file_type"]["checkpoint_files"]
    errors, _ = _run(cfg)
    assert any("checkpoint_files" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.8 background in primary metrics
# ---------------------------------------------------------------------------

def test_background_must_be_excluded_from_primary_macro() -> None:
    cfg = _load_good()
    cfg["metrics"]["primary_metrics"]["background_included_in_macro_average"] = True
    errors, _ = _run(cfg)
    assert any("background_included_in_macro_average" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.9 class weight mean excludes background
# ---------------------------------------------------------------------------

def test_class_weight_mean_must_exclude_background() -> None:
    cfg = _load_good()
    cfg["dataset"]["class_weight_formula"]["background_excluded_from_mean"] = False
    errors, _ = _run(cfg)
    assert any("background_excluded_from_mean" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.10 TEST=0
# ---------------------------------------------------------------------------

def test_test_must_be_denied_top_level() -> None:
    cfg = _load_good()
    cfg["test_access_policy"]["this_run_loads_test"] = True
    errors, _ = _run(cfg)
    assert any("TEST access violation" in e for e in errors), errors


def test_test_must_be_denied_dataset() -> None:
    cfg = _load_good()
    cfg["dataset"]["test_access_policy"]["load_test_in_mini"] = True
    errors, _ = _run(cfg)
    assert any("TEST access violation" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.11 SegFormer DEFERRED
# ---------------------------------------------------------------------------

def test_segformer_must_be_deferred() -> None:
    cfg = _load_good()
    cfg["candidates"][3]["role"] = "new_candidate"
    errors, _ = _run(cfg)
    assert any("SegFormer" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.12 architectures complete freeze
# ---------------------------------------------------------------------------

def test_resunet_architecture_freeze_required() -> None:
    cfg = _load_good()
    del cfg["candidates"][1]["architecture_freeze"]
    errors, _ = _run(cfg)
    assert any("architecture_freeze" in e for e in errors), errors


def test_deeplabv3plus_aspp_atrous_rates_required() -> None:
    cfg = _load_good()
    fp = cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
    aspp_block = next(b for b in fp if b["name"] == "aspp")
    aspp_block["aspp_settings"]["atrous_rates"] = []
    errors, _ = _run(cfg)
    assert any("atrous_rates must be a list of >=3 rates" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.13 worst-subject floor wording
# ---------------------------------------------------------------------------

def test_worst_subject_floor_must_not_claim_50pct_of_b02() -> None:
    cfg = _load_good()
    cfg["worst_subject_guardrail"]["policy_note"] = (
        "0.20 is approximately 50% of the B02 baseline of 0.205644."
    )
    errors, _ = _run(cfg)
    assert any("50% of B02" in e or "50% of the B02" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.14 B01 SHA
# ---------------------------------------------------------------------------

def test_b01_a06_sha_mismatch_caught() -> None:
    cfg = _load_good()
    cfg["b01_a06_split_sha256_expected"] = "0" * 64
    errors, _ = _run(cfg)
    assert any("b01_a06_split_sha256_expected mismatch" in e for e in errors), errors


def test_b01_manifest_sha_mismatch_caught() -> None:
    cfg = _load_good()
    cfg["b01_freeze_manifest_core_sha256_expected"] = "0" * 64
    errors, _ = _run(cfg)
    assert any("b01_freeze_manifest_core_sha256_expected mismatch" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R02.15 feasibility threshold arithmetic
# ---------------------------------------------------------------------------

def test_feasibility_threshold_must_be_355644() -> None:
    cfg = _load_good()
    cfg["feasibility_gate"]["effective_threshold"] = 0.5
    errors, _ = _run(cfg)
    assert any("effective_threshold" in e for e in errors), errors


def test_feasibility_threshold_comparison_must_be_geq() -> None:
    cfg = _load_good()
    cfg["feasibility_gate"]["threshold_comparison_operator"] = ">"
    errors, _ = _run(cfg)
    assert any("threshold_comparison_operator" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03.7 Residual Add shape/channel consistency
# ---------------------------------------------------------------------------

def test_residual_shortcut_must_be_conv2d_not_identity() -> None:
    cfg = _load_good()
    # Replace ResUNet enc1 shortcut with an Identity op
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"] = {"op": "Identity", "kind": "identity"}
    errors, _ = _run(cfg)
    assert any("shortcut: must be a Conv2d operation" in e for e in errors), errors


def test_residual_shortcut_must_be_1x1_kernel() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"]["kernel_size"] = 3  # wrong: must be 1
    errors, _ = _run(cfg)
    assert any("kernel_size must be 1" in e for e in errors), errors


def test_residual_shortcut_padding_must_be_zero() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"]["padding"] = 1
    errors, _ = _run(cfg)
    assert any("padding must be 0" in e for e in errors), errors


def test_residual_shortcut_stride_must_be_one() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"]["stride"] = 2
    errors, _ = _run(cfg)
    assert any("stride must be 1" in e for e in errors), errors


def test_residual_shortcut_bias_must_be_true() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"]["bias"] = False
    errors, _ = _run(cfg)
    assert any("bias must be true" in e for e in errors), errors


def test_residual_add_channels_mismatch_caught() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["main_output_channels"] = 16
    enc1["shortcut_output_channels"] = 32  # mismatch
    errors, _ = _run(cfg)
    assert any("main_output_channels (16) == shortcut_output_channels (32)" in e for e in errors), errors


def test_residual_add_shape_mismatch_caught() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["main_output_shape"] = [192, 84]
    enc1["shortcut_output_shape"] = [96, 42]  # mismatch
    errors, _ = _run(cfg)
    assert any("main_output_shape" in e and "shortcut_output_shape" in e for e in errors), errors


def test_residual_shortcut_in_channels_must_match_block_input() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][1]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1_resblock"
    )
    enc1["shortcut"]["in_channels"] = 2  # wrong: should be 1
    errors, _ = _run(cfg)
    assert any("must match block input channels" in e for e in errors), errors


def test_residual_projection_text_must_not_say_identity_or_extra() -> None:
    cfg = _load_good()
    cfg["candidates"][1]["architecture_freeze"]["residual_projection"] = (
        "identity with extra conv for now"
    )
    errors, _ = _run(cfg)
    assert any("must not mention" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03.8 Concat result_channels consistency
# ---------------------------------------------------------------------------

def test_concat_result_channels_must_ge_with_channels() -> None:
    cfg = _load_good()
    # SmallUNet dec1 Concat: result=96, with=32
    dec1 = next(
        b for b in cfg["candidates"][0]["architecture_freeze"]["forward_plan"]
        if b["name"] == "dec1"
    )
    for layer in dec1["ops"]:
        if layer.get("op") == "Concat":
            layer["result_channels"] = 16  # wrong: < with_channels
    errors, _ = _run(cfg)
    assert any("Concat result_channels" in e and "must be >=" in e for e in errors), errors


def test_aspp_concat_result_must_equal_sum_of_branches() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    aspp["concat"]["result_channels"] = 999  # wrong: must equal 80
    errors, _ = _run(cfg)
    assert any("result_channels (999) must equal sum" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03.9 ASPP branch output / post_concat consistency
# ---------------------------------------------------------------------------

def test_aspp_branch_output_channels_must_match_per_branch_setting() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    aspp["branches"][1]["output_channels"] = 32  # wrong: should be 16
    errors, _ = _run(cfg)
    assert any("output_channels=32 != aspp_settings.output_channels_per_branch=16" in e for e in errors), errors


def test_aspp_expected_post_concat_must_match_computed() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    aspp["aspp_settings"]["expected_post_concat_input_channels"] = 999
    errors, _ = _run(cfg)
    assert any("expected_post_concat_input_channels" in e for e in errors), errors


def test_aspp_post_concat_in_channels_must_match_expected() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    aspp["post_concat"][0]["in_channels"] = 999
    errors, _ = _run(cfg)
    assert any("in_channels=999 != expected 96" in e for e in errors), errors


def test_aspp_atrous_dilation_must_match_branch_rate() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    # Branch 1 has rate 3, but change its Conv2d dilation to 6
    for layer in aspp["branches"][1]["ops"]:
        if layer.get("op") == "Conv2d":
            layer["dilation"] = 6
    errors, _ = _run(cfg)
    assert any("dilation=6 != atrous_rate=3" in e for e in errors), errors


def test_aspp_atrous_padding_must_match_branch_rate() -> None:
    cfg = _load_good()
    aspp = next(
        b for b in cfg["candidates"][2]["architecture_freeze"]["forward_plan"]
        if b["name"] == "aspp"
    )
    for layer in aspp["branches"][1]["ops"]:
        if layer.get("op") == "Conv2d":
            layer["padding"] = 1
    errors, _ = _run(cfg)
    assert any("padding=1 != atrous_rate=3" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03.10 depthwise groups consistency
# ---------------------------------------------------------------------------

def test_depthwise_groups_must_equal_in_channels_caught() -> None:
    cfg = _load_good()
    # Add a fake depthwise layer to a non-ASPP block in SmallUNet
    enc1 = next(
        b for b in cfg["candidates"][0]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1"
    )
    enc1["ops"].append(
        {
            "op": "Conv2d",
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
            "groups": 16,  # depthwise: not allowed in R03 Option A scope
        }
    )
    errors, _ = _run(cfg)
    assert any("depthwise Conv2d" in e for e in errors), errors


def test_arbitrary_grouped_conv_caught() -> None:
    cfg = _load_good()
    enc1 = next(
        b for b in cfg["candidates"][0]["architecture_freeze"]["forward_plan"]
        if b["name"] == "enc1"
    )
    enc1["ops"].append(
        {
            "op": "Conv2d",
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
            "groups": 4,  # neither 1 nor in_channels
        }
    )
    errors, _ = _run(cfg)
    assert any("grouped Conv2d" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03.11 exact parameter count consistency
# ---------------------------------------------------------------------------

def test_exact_parameter_count_must_be_present() -> None:
    cfg = _load_good()
    del cfg["candidates"][0]["exact_parameter_count"]
    errors, _ = _run(cfg)
    assert any("exact_parameter_count field missing" in e for e in errors), errors


def test_exact_parameter_count_must_match_computed() -> None:
    cfg = _load_good()
    cfg["candidates"][0]["exact_parameter_count"] = 99999  # wrong
    errors, _ = _run(cfg)
    assert any("exact_parameter_count=99999 != computed 118121" in e for e in errors), errors


def test_resunet_exact_parameter_count_must_be_120809() -> None:
    cfg = _load_good()
    cfg["candidates"][1]["exact_parameter_count"] = 99999
    errors, _ = _run(cfg)
    assert any("exact_parameter_count=99999 != computed 120809" in e for e in errors), errors


def test_deeplab_exact_parameter_count_must_be_53449() -> None:
    cfg = _load_good()
    cfg["candidates"][2]["exact_parameter_count"] = 99999
    errors, _ = _run(cfg)
    assert any("exact_parameter_count=99999 != computed 53449" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R03 DeepLab Option A enforcement
# ---------------------------------------------------------------------------

def test_deeplab_must_declare_option_a_or_plain() -> None:
    cfg = _load_good()
    cfg["candidates"][2]["architecture_freeze"]["variant"] = "option_B_depthwise"
    errors, _ = _run(cfg)
    assert any("must declare 'option_A' or 'plain atrous'" in e for e in errors), errors
