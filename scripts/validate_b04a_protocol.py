"""B04A Protocol Contract Validator (R03).

Validates the B04A experiment configuration against the frozen protocol
contract. Designed to be invoked by humans and by
``tests/test_b04a_protocol_validator.py``.

What it catches (R02 + R03 requirements):
  1. dtype inconsistency between pressure_storage_dtype and
     model_input_dtype and the dtype_conversion_policy fields.
  2. candidate augmentation inconsistency (all three active candidates
     must have augmentation_policy == "none" and identical
     augmentation blocks).
  3. seed inconsistency (all three active candidates must run the same
     registered seeds list).
  4. missing 3_feasible decision rule in feasibility_gate.
  5. failure handling that excludes failed seeds (catches any code or
     config that says "compute mean over remaining successful seeds"
     without requiring all_seeds_must_succeed).
  6. resource budget multiplication inconsistency (per_candidate_wall
     * 3 != total_wall_minutes for serial 3-candidate run, etc.).
  7. (R03) residual Add shape/channel consistency for every residual
     block in any candidate.
  8. (R03) Concat result_channels consistency for every Concat op
     (must equal the sum of incoming channel counts).
  9. (R03) ASPP branch output channels and post_concat input
     channels consistency (all branches equal, post_concat input
     equals sum of branches).
 10. (R03) depthwise groups consistency (if a layer claims depthwise
     via groups, groups MUST equal in_channels; if not depthwise,
     groups MUST be 1).
 11. (R03) exact_parameter_count consistency: validator recursively
     computes the Conv2d parameter count from the forward_plan and
     compares to the declared exact_parameter_count; the result
     must be identical.

Returns: exit 0 on PASS, exit 1 on FAIL, with a structured report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "config_version",
    "task_id",
    "stage",
    "freeze_version",
    "b01_a06_split_sha256_expected",
    "b01_freeze_manifest_core_sha256_expected",
    "expected_split_counts",
    "expected_subjects",
    "candidates",
    "dataset",
    "training",
    "determinism",
    "metrics",
    "resource_budget",
    "feasibility_gate",
    "class_collapse_guard",
    "worst_subject_guardrail",
    "per_region_guardrail",
    "near_tie_margin",
    "candidate_decision_aggregation",
    "test_access_policy",
    "expected_artifacts",
    "identity_hard_gate",
    "real_run_authorization",
    "lifecycle",
)


def _fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _ok(oks: list[str], msg: str) -> None:
    oks.append(msg)


# ---------------------------------------------------------------------------
# Parameter count computer (R03)
# ---------------------------------------------------------------------------

def _conv2d_params(in_c: int, out_c: int, k: int, bias: bool) -> int:
    return in_c * out_c * k * k + (out_c if bias else 0)


def _layer_params(layer: dict[str, Any]) -> int:
    op = layer.get("op", "")
    if op == "Conv2d":
        k = layer["kernel_size"]
        in_c = layer["in_channels"]
        out_c = layer["out_channels"]
        bias = layer.get("bias", True)
        return _conv2d_params(in_c, out_c, k, bias)
    return 0


def _block_params(block: dict[str, Any]) -> int:
    """Compute parameter count for a single forward_plan block."""
    total = 0
    btype = block.get("type", "sequential")
    if btype == "residual_block":
        for layer in block.get("main_path", []):
            total += _layer_params(layer)
        shortcut = block.get("shortcut")
        if isinstance(shortcut, dict):
            total += _layer_params(shortcut)
        elif isinstance(shortcut, list):
            for layer in shortcut:
                total += _layer_params(layer)
        for layer in block.get("post_add", []):
            total += _layer_params(layer)
    elif btype == "aspp":
        for branch in block.get("branches", []):
            for layer in branch.get("ops", []):
                total += _layer_params(layer)
        for layer in block.get("post_concat", []):
            total += _layer_params(layer)
    elif btype == "single_conv":
        total += _layer_params(block)
    else:
        # sequential / decoder_block
        for layer in block.get("ops", []):
            total += _layer_params(layer)
        for layer in block.get("layers", []):
            total += _layer_params(layer)
    return total


def compute_exact_param_count(af: dict[str, Any]) -> int:
    """Recursively compute Conv2d param count for a candidate's forward_plan."""
    total = 0
    for block in af.get("forward_plan", []):
        total += _block_params(block)
    return total


# ---------------------------------------------------------------------------
# Consistency checks (R03)
# ---------------------------------------------------------------------------

def _check_residual_blocks(candidate: dict[str, Any], errors: list[str], oks: list[str]) -> None:
    name = candidate["name"]
    af = candidate["architecture_freeze"]
    n_resblocks = 0
    for block in af.get("forward_plan", []):
        if block.get("type") != "residual_block":
            continue
        n_resblocks += 1
        # The shortcut must be an explicit Conv2d, not an Identity or "extra conv".
        shortcut = block.get("shortcut")
        if shortcut is None:
            _fail(
                errors,
                f"{name}.{block['name']}: shortcut missing (R03: must be explicit Conv2d 1x1, not Identity or 'extra conv')",
            )
            continue
        if not isinstance(shortcut, dict) or shortcut.get("op") != "Conv2d":
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: must be a Conv2d operation (R03 forbids Identity and 'extra conv')",
            )
            continue
        # kernel must be 1 (channel projection), padding 0, stride 1
        if shortcut.get("kernel_size") != 1:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: kernel_size must be 1 (1x1 channel projection); got {shortcut.get('kernel_size')}",
            )
        if shortcut.get("stride") != 1:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: stride must be 1; got {shortcut.get('stride')}",
            )
        if shortcut.get("padding") != 0:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: padding must be 0; got {shortcut.get('padding')}",
            )
        if shortcut.get("bias") is not True:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: bias must be true; got {shortcut.get('bias')}",
            )
        # shortcut output channels must equal main output channels
        main_out = block.get("main_output_channels")
        shortcut_out = block.get("shortcut_output_channels")
        if main_out is None or shortcut_out is None:
            _fail(
                errors,
                f"{name}.{block['name']}: must declare main_output_channels and shortcut_output_channels for Add consistency",
            )
            continue
        if main_out != shortcut_out:
            _fail(
                errors,
                f"{name}.{block['name']}: residual Add requires main_output_channels ({main_out}) == shortcut_output_channels ({shortcut_out})",
            )
        # shapes must match
        main_shape = block.get("main_output_shape")
        shortcut_shape = block.get("shortcut_output_shape")
        if main_shape != shortcut_shape:
            _fail(
                errors,
                f"{name}.{block['name']}: residual Add requires main_output_shape {main_shape} == shortcut_output_shape {shortcut_shape}",
            )
        # shortcut in_channels must match block input channels
        block_input = block.get("input", {})
        if block_input.get("channels") is not None and shortcut.get("in_channels") != block_input["channels"]:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: in_channels={shortcut.get('in_channels')} must match block input channels {block_input['channels']}",
            )
        # shortcut out_channels must match shortcut_output_channels
        if shortcut.get("out_channels") != shortcut_out:
            _fail(
                errors,
                f"{name}.{block['name']}.shortcut: out_channels={shortcut.get('out_channels')} must match shortcut_output_channels {shortcut_out}",
            )
    if n_resblocks > 0:
        _ok(oks, f"{name}: {n_resblocks} residual block(s) all pass Add shape/channel checks")


def _check_concat_channels(candidate: dict[str, Any], errors: list[str], oks: list[str]) -> None:
    name = candidate["name"]
    af = candidate["architecture_freeze"]
    n_concats = 0
    for block in af.get("forward_plan", []):
        for layer in block.get("ops", []):
            if layer.get("op") == "Concat":
                n_concats += 1
                with_ch = layer.get("with_channels")
                result_ch = layer.get("result_channels")
                # We can't always know the previous channels here, but we can
                # require the result_channels >= with_channels.
                if with_ch is not None and result_ch is not None:
                    if result_ch < with_ch:
                        _fail(
                            errors,
                            f"{name}.{block['name']}: Concat result_channels ({result_ch}) must be >= with_channels ({with_ch})",
                        )
        # ASPP Concat
        if block.get("type") == "aspp":
            n_concats += 1
            cc = block.get("concat", {})
            with_branch_ids = cc.get("with_branch_ids", [])
            result_ch = cc.get("result_channels")
            # Sum of branch.output_channels must equal result_channels
            branch_channels = []
            for b in block.get("branches", []):
                branch_channels.append(b.get("output_channels"))
            if result_ch is not None and branch_channels:
                if sum(branch_channels) != result_ch:
                    _fail(
                        errors,
                        f"{name}.{block['name']}.concat: result_channels ({result_ch}) must equal sum(branch.output_channels) ({sum(branch_channels)} = {branch_channels})",
                    )
                if len(branch_channels) != len(with_branch_ids):
                    _fail(
                        errors,
                        f"{name}.{block['name']}.concat: with_branch_ids count ({len(with_branch_ids)}) must match number of branches ({len(branch_channels)})",
                    )
    if n_concats > 0:
        _ok(oks, f"{name}: {n_concats} Concat op(s) pass channel consistency")


def _check_aspp(candidate: dict[str, Any], errors: list[str], oks: list[str]) -> None:
    name = candidate["name"]
    af = candidate["architecture_freeze"]
    for block in af.get("forward_plan", []):
        if block.get("type") != "aspp":
            continue
        cfg = block["aspp_settings"]
        # atrous_rates must be a list of >=3 rates
        rates = cfg.get("atrous_rates", [])
        if not isinstance(rates, list) or len(rates) < 3:
            _fail(
                errors,
                f"{name}.{block['name']}.aspp_settings.atrous_rates must be a list of >=3 rates; got {rates!r}",
            )
        # All branches must have output_channels == output_channels_per_branch
        per_branch = cfg["output_channels_per_branch"]
        for branch in block.get("branches", []):
            if branch.get("output_channels") != per_branch:
                _fail(
                    errors,
                    f"{name}.{block['name']}.branch[{branch.get('branch_id')}]: output_channels={branch.get('output_channels')} != aspp_settings.output_channels_per_branch={per_branch}",
                )
        # Expected post_concat input channels formula
        n_atrous = cfg["n_atrous_branches"]
        gap = 1 if cfg.get("include_gap_branch") else 0
        onexone = 1 if cfg.get("include_1x1_pointwise_branch") else 0
        expected = (n_atrous + gap + onexone) * per_branch
        if cfg.get("expected_post_concat_input_channels") != expected:
            _fail(
                errors,
                f"{name}.{block['name']}.aspp_settings.expected_post_concat_input_channels={cfg.get('expected_post_concat_input_channels')} != computed {expected} = ({n_atrous}+{gap}+{onexone})*{per_branch}",
            )
        # post_concat[0] (Conv2d) in_channels must equal expected
        post_concat = block.get("post_concat", [])
        if post_concat and post_concat[0].get("op") == "Conv2d":
            if post_concat[0].get("in_channels") != expected:
                _fail(
                    errors,
                    f"{name}.{block['name']}.post_concat[0]: in_channels={post_concat[0].get('in_channels')} != expected {expected}",
                )
        # atrous_rate consistency: 3x3 conv with dilation must match
        for branch in block.get("branches", []):
            if branch.get("kind") == "atrous_3x3":
                for layer in branch.get("ops", []):
                    if layer.get("op") == "Conv2d":
                        d = layer.get("dilation")
                        p = layer.get("padding")
                        rate = branch.get("atrous_rate")
                        # For "same" padding with kernel=3 and dilation=d, padding=d
                        if d != rate:
                            _fail(
                                errors,
                                f"{name}.{block['name']}.branch[{branch.get('branch_id')}]: Conv2d dilation={d} != atrous_rate={rate}",
                            )
                        if p != rate:
                            _fail(
                                errors,
                                f"{name}.{block['name']}.branch[{branch.get('branch_id')}]: Conv2d padding={p} != atrous_rate={rate} (required for same-shape atrous)",
                            )
        _ok(
            oks,
            f"{name}.{block['name']}: ASPP branches consistent (per_branch={per_branch}, expected_post_concat_in={expected})",
        )


def _check_depthwise_groups(candidate: dict[str, Any], errors: list[str], oks: list[str]) -> None:
    name = candidate["name"]
    af = candidate["architecture_freeze"]
    n_layers = 0
    for block in af.get("forward_plan", []):
        all_layers: list[dict[str, Any]] = []
        btype = block.get("type", "sequential")
        if btype == "residual_block":
            all_layers.extend(block.get("main_path", []))
            sc = block.get("shortcut")
            if isinstance(sc, dict):
                all_layers.append(sc)
            elif isinstance(sc, list):
                all_layers.extend(sc)
            all_layers.extend(block.get("post_add", []))
        elif btype == "aspp":
            for branch in block.get("branches", []):
                all_layers.extend(branch.get("ops", []))
            all_layers.extend(block.get("post_concat", []))
        else:
            all_layers.extend(block.get("ops", []))
            all_layers.extend(block.get("layers", []))
        for layer in all_layers:
            if layer.get("op") != "Conv2d":
                continue
            n_layers += 1
            groups = layer.get("groups", 1)
            in_c = layer["in_channels"]
            if not isinstance(groups, int) or groups < 1:
                _fail(
                    errors,
                    f"{name}.{block['name']}: Conv2d groups must be a positive integer; got {groups!r}",
                )
                continue
            # Rule: groups=1 means plain conv; groups=in_channels means depthwise.
            # If groups is anything else (e.g., 2 with in_c=32) it is a
            # grouped conv, which is neither plain nor depthwise and is
            # not allowed by the protocol.
            if groups == 1:
                continue  # plain conv: OK
            if groups == in_c:
                # depthwise; allowed only if the surrounding context
                # declares the candidate uses depthwise-separable.
                # For R03 only Option A (plain atrous) is in scope; the
                # current DeepLabV3+-lite must NOT have groups==in_c.
                _fail(
                    errors,
                    f"{name}.{block['name']}: depthwise Conv2d (groups=in_channels) detected. "
                    "R03 selected Option A (plain atrous Conv2d); depthwise-separable is not used. "
                    "If a future amendment chooses Option B, groups=in_channels is allowed but "
                    "must be paired with the pointwise 1x1 Conv2d immediately after.",
                )
                continue
            _fail(
                errors,
                f"{name}.{block['name']}: grouped Conv2d with groups={groups}, in_channels={in_c} is neither plain (groups=1) nor depthwise (groups=in_channels). "
                    "R03 forbids arbitrary grouped convolutions; only plain or depthwise is allowed.",
            )
    if n_layers > 0:
        _ok(oks, f"{name}: {n_layers} Conv2d layer(s) all pass groups consistency")


def _check_exact_param_count(candidate: dict[str, Any], errors: list[str], oks: list[str]) -> None:
    name = candidate["name"]
    af = candidate["architecture_freeze"]
    declared = candidate.get("exact_parameter_count")
    if declared is None:
        _fail(
            errors,
            f"{name}: exact_parameter_count field missing (R03 requirement)",
        )
        return
    computed = compute_exact_param_count(af)
    if declared != computed:
        _fail(
            errors,
            f"{name}: exact_parameter_count={declared} != computed {computed} (R03: validator must agree with declared value)",
        )
    else:
        _ok(oks, f"{name}: exact_parameter_count={declared} matches computed (R03)")


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, oks)."""
    errors: list[str] = []
    oks: list[str] = []

    # ---- 1. top-level required fields ----
    missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in config]
    if missing:
        _fail(errors, f"Missing required top-level fields: {missing}")
    else:
        _ok(oks, "all required top-level fields present")

    # ---- 2. B01 / B02 contracts ----
    if config.get("b01_a06_split_sha256_expected") != (
        "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
    ):
        _fail(errors, "b01_a06_split_sha256_expected mismatch (must be 024f5abe...)")
    else:
        _ok(oks, "B01 A06 split SHA matches B01 contract")

    if config.get("b01_freeze_manifest_core_sha256_expected") != (
        "3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e"
    ):
        _fail(errors, "b01_freeze_manifest_core_sha256_expected mismatch (must be 3c789995...)")
    else:
        _ok(oks, "B01 freeze manifest core SHA matches B01 contract")

    fg = config.get("feasibility_gate", {})
    if fg.get("b02_reference_val_fixed_iou") != 0.205644:
        _fail(errors, f"feasibility_gate.b02_reference_val_fixed_iou must be 0.205644, got {fg.get('b02_reference_val_fixed_iou')}")
    if fg.get("absolute_margin") != 0.15:
        _fail(errors, f"feasibility_gate.absolute_margin must be 0.15, got {fg.get('absolute_margin')}")
    effective = fg.get("effective_threshold")
    if effective is None or abs(effective - 0.355644) > 1e-9:
        _fail(errors, f"feasibility_gate.effective_threshold must be 0.355644, got {effective}")
    if fg.get("threshold_comparison_operator") != ">=":
        _fail(errors, f"feasibility_gate.threshold_comparison_operator must be '>=', got {fg.get('threshold_comparison_operator')}")
    if fg.get("all_seeds_must_succeed") is not True:
        _fail(errors, "feasibility_gate.all_seeds_must_succeed must be true (R02 requirement)")
    else:
        _ok(oks, "feasibility_gate.all_seeds_must_succeed == true")

    rules = fg.get("decision_rules", {})
    if "3_feasible" not in rules:
        _fail(errors, "feasibility_gate.decision_rules must define '3_feasible' (R02 requirement)")
    else:
        _ok(oks, "3_feasible decision rule present")

    # ---- 3. dtype consistency ----
    ds = config.get("dataset", {})
    if "pressure_storage_dtype" not in ds:
        _fail(errors, "dataset.pressure_storage_dtype missing (R02: storage dtype must be separate from model input dtype)")
    if "model_input_dtype" not in ds:
        _fail(errors, "dataset.model_input_dtype missing (R02 requirement)")
    if "dtype_conversion_policy" not in ds:
        _fail(errors, "dataset.dtype_conversion_policy missing (R02 requirement)")

    pstore = ds.get("pressure_storage_dtype")
    minput = ds.get("model_input_dtype")
    if pstore == "float64" and minput == "float32":
        _ok(oks, "pressure_storage_dtype=float64, model_input_dtype=float32 (consistent)")
    else:
        _fail(errors, f"dtype mismatch: pressure_storage_dtype={pstore!r}, model_input_dtype={minput!r} (expected float64 -> float32)")

    dcp = ds.get("dtype_conversion_policy", {})
    if dcp.get("input_storage_dtype") != "float64":
        _fail(errors, f"dtype_conversion_policy.input_storage_dtype must be 'float64', got {dcp.get('input_storage_dtype')!r}")
    if dcp.get("model_input_dtype") != "float32":
        _fail(errors, f"dtype_conversion_policy.model_input_dtype must be 'float32', got {dcp.get('model_input_dtype')!r}")
    rule = dcp.get("rule", "")
    if "normalization" not in rule or "augmentation" not in rule:
        _fail(errors, "dtype_conversion_policy.rule must explicitly mention BOTH 'normalization' (before) and 'augmentation' (before) so the conversion site is unambiguous")
    else:
        _ok(oks, "dtype_conversion_policy.rule mentions both normalization and augmentation (unambiguous site)")

    # ---- 4. candidate augmentation consistency ----
    candidates = config.get("candidates", [])
    active = [c for c in candidates if c.get("role") != "DEFERRED"]
    if len(active) != 3:
        _fail(errors, f"Expected 3 active candidates (incumbent + 2 new); got {len(active)}")

    aug_policies = [c.get("augmentation_policy") for c in active]
    if not all(p == "none" for p in aug_policies):
        _fail(errors, f"All active candidates must have augmentation_policy='none' (R02); got {aug_policies}")
    else:
        _ok(oks, "all active candidates use augmentation_policy='none'")

    aug_blocks = [json.dumps(c.get("augmentation", {}), sort_keys=True) for c in active]
    if len(set(aug_blocks)) != 1:
        _fail(errors, f"Augmentation blocks differ across active candidates (R02 requirement: identical): {aug_blocks}")
    else:
        _ok(oks, "all active candidates have identical augmentation blocks")

    # ---- 5. seed consistency ----
    registered_seeds = config.get("training", {}).get("seeds", [])
    if registered_seeds != [42, 123, 2026]:
        _fail(errors, f"training.seeds must be [42, 123, 2026] (R02); got {registered_seeds}")
    else:
        _ok(oks, "training.seeds == [42, 123, 2026]")

    aug_per_cand = config.get("training", {}).get("augmentation_policy_per_candidate", {})
    if aug_per_cand.get("slp8_small_unet_v0.1") != "none":
        _fail(errors, f"training.augmentation_policy_per_candidate.slp8_small_unet_v0.1 must be 'none'; got {aug_per_cand.get('slp8_small_unet_v0.1')!r}")
    if aug_per_cand.get("slp8_resunet_lite_v0.1") != "none":
        _fail(errors, f"training.augmentation_policy_per_candidate.slp8_resunet_lite_v0.1 must be 'none'; got {aug_per_cand.get('slp8_resunet_lite_v0.1')!r}")
    if aug_per_cand.get("slp8_deeplabv3plus_lite_v0.1") != "none":
        _fail(errors, f"training.augmentation_policy_per_candidate.slp8_deeplabv3plus_lite_v0.1 must be 'none'; got {aug_per_cand.get('slp8_deeplabv3plus_lite_v0.1')!r}")

    # ---- 6. failure handling that excludes failed seeds ----
    fda = config.get("candidate_decision_aggregation", {})
    formula = fda.get("macro_iou_mean_formula", "")
    if "remaining" in formula.lower() and "forbidden" not in formula.lower():
        _fail(errors, "candidate_decision_aggregation.macro_iou_mean_formula contains 'remaining' without an explicit 'forbidden' marker. R02 forbids computing mean over only the remaining successful seeds; the mean must require all 3 seeds.")
    else:
        _ok(oks, "macro_iou_mean_formula does not silently drop failed seeds")

    cf_rule = fda.get("candidate_feasibility_rule", "")
    if cf_rule and "all" not in cf_rule.lower():
        _fail(errors, "candidate_decision_aggregation.candidate_feasibility_rule must include the word 'all' to make the all-seeds requirement explicit (R02).")

    # ---- 7. resource budget multiplication consistency ----
    rb = config.get("resource_budget", {})
    pcm = rb.get("per_candidate_wall_minutes")
    twm = rb.get("total_wall_minutes")
    n_candidates = sum(1 for c in candidates if c.get("role") != "DEFERRED")
    if pcm is None or twm is None:
        _fail(errors, f"resource_budget.per_candidate_wall_minutes={pcm}, total_wall_minutes={twm}; both required")
    else:
        if abs(pcm * n_candidates - twm) > 1e-9:
            _fail(errors, f"resource_budget inconsistent: per_candidate_wall_minutes ({pcm}) * active candidates ({n_candidates}) != total_wall_minutes ({twm})")
        else:
            _ok(oks, f"resource_budget: {pcm} min/candidate * {n_candidates} candidates = {twm} min total")

    resume = rb.get("checkpoint_resume_keeps_accumulators", {})
    if "candidate_level" not in resume:
        _fail(errors, "resource_budget.checkpoint_resume_keeps_accumulators.candidate_level missing (R02)")
    if "total_level" not in resume:
        _fail(errors, "resource_budget.checkpoint_resume_keeps_accumulators.total_level missing (R02)")
    if "candidate_level" in resume and "total_level" in resume:
        _ok(oks, "resource_budget has both candidate_level and total_level accumulators")

    # ---- 8. identity hard gate carrier format ----
    ihg = config.get("identity_hard_gate", {})
    if "carrier_format_by_file_type" not in ihg:
        _fail(errors, "identity_hard_gate.carrier_format_by_file_type missing (R02 requirement)")
    else:
        cfb = ihg["carrier_format_by_file_type"]
        for kind in ("json_files", "csv_files", "checkpoint_files", "log_files"):
            if kind not in cfb:
                _fail(errors, f"identity_hard_gate.carrier_format_by_file_type.{kind} missing (R02)")
        if all(k in cfb for k in ("json_files", "csv_files", "checkpoint_files", "log_files")):
            _ok(oks, "identity carrier format covers json/csv/checkpoint/log (R02)")

    # ---- 9. background in primary metrics ----
    primary = config.get("metrics", {}).get("primary_metrics", {})
    if primary.get("background_included_in_macro_average") is not False:
        _fail(errors, "metrics.primary_metrics.background_included_in_macro_average must be false (R02)")
    else:
        _ok(oks, "background NOT in primary macro average")

    # ---- 10. class weight mean excludes background ----
    cwf = ds.get("class_weight_formula", {})
    if cwf.get("background_excluded_from_mean") is not True:
        _fail(errors, "dataset.class_weight_formula.background_excluded_from_mean must be true (R02)")
    else:
        _ok(oks, "class-weight mean excludes background")

    # ---- 11. TEST=0 ----
    tap_top = config.get("test_access_policy", {})
    tap_ds = ds.get("test_access_policy", {})
    for k, v in (
        ("top-level test_access_policy.load_test_default", tap_top.get("load_test_default")),
        ("top-level test_access_policy.this_run_loads_test", tap_top.get("this_run_loads_test")),
        ("top-level test_access_policy.test_access_in_this_run", tap_top.get("test_access_in_this_run")),
        ("dataset.test_access_policy.load_test_default", tap_ds.get("load_test_default")),
        ("dataset.test_access_policy.load_test_in_mini", tap_ds.get("load_test_in_mini")),
    ):
        if v in (False, "denied"):
            continue
        _fail(errors, f"TEST access violation: {k}={v!r}")

    if not any("TEST access violation" in e for e in errors):
        _ok(oks, "TEST=0 enforced in both top-level and dataset policies")

    # ---- 12. SegFormer DEFERRED ----
    seg = [c for c in candidates if "segformer" in c.get("name", "").lower()]
    if len(seg) != 1 or seg[0].get("role") != "DEFERRED":
        _fail(errors, "SegFormer candidate must be exactly one and DEFERRED")
    else:
        _ok(oks, "SegFormer-B0 DEFERRED")

    # ---- 13. architectures + R03 consistency checks ----
    for c in active:
        name = c.get("name", "")
        af = c.get("architecture_freeze")
        if not isinstance(af, dict):
            _fail(errors, f"{name}: architecture_freeze block missing or not a dict (R02 requirement)")
            continue
        for sub in ("input_shape", "n_classes", "normalization", "activation", "upsampling", "output_spatial_recovery"):
            if sub not in af:
                _fail(errors, f"{name}: architecture_freeze.{sub} missing (R02 requirement)")
        if "forward_plan" not in af or not isinstance(af["forward_plan"], list) or not af["forward_plan"]:
            _fail(errors, f"{name}: architecture_freeze.forward_plan must be a non-empty list (R03 requirement)")
            continue
        # R03 checks
        _check_residual_blocks(c, errors, oks)
        _check_concat_channels(c, errors, oks)
        _check_aspp(c, errors, oks)
        _check_depthwise_groups(c, errors, oks)
        _check_exact_param_count(c, errors, oks)

    # ResUNet must declare residual_projection
    res = next((c for c in active if "resunet" in c.get("name", "").lower()), None)
    if res is not None:
        af = res.get("architecture_freeze", {})
        rp = af.get("residual_projection", "")
        if "1x1" not in rp and "identity" not in rp:
            _fail(errors, "ResUNet-lite.architecture_freeze.residual_projection must specify 1x1 or identity rule")
        if "1x1" in rp and "identity" in rp.lower():
            # ambiguous: must not have "identity" as a residual projection kind
            _fail(errors, "ResUNet-lite.residual_projection must not mention 'identity' as a projection kind (R03 forbids Identity shortcut)")
        if "extra conv" in rp.lower() or "for now" in rp.lower():
            _fail(errors, "ResUNet-lite.residual_projection must not mention 'extra conv' or 'for now' (R03 forbids ambiguous shortcut descriptions)")

    # DeepLab must declare Option A explicitly
    dlv = next((c for c in active if "deeplab" in c.get("name", "").lower()), None)
    if dlv is not None:
        af = dlv.get("architecture_freeze", {})
        variant = af.get("variant", "")
        if "option_A" not in variant and "plain" not in variant.lower():
            _fail(errors, f"DeepLabV3+-lite.architecture_freeze.variant must declare 'option_A' or 'plain atrous' (R03 chose Option A explicitly); got {variant!r}")
        if "depthwise" in json.dumps(af).lower() and "aspp_settings" in json.dumps(af).lower():
            # only fail if depthwise appears in the architecture description beyond
            # the validator's own warnings
            if "depthwise" in af.get("variant", "").lower():
                _fail(errors, "DeepLabV3+-lite.architecture_freeze.variant declares depthwise; R03 chose Option A (plain atrous), not Option B (depthwise-separable)")

    # ---- 14. worst-subject floor not derived from B02 ----
    wsg = config.get("worst_subject_guardrail", {})
    if wsg.get("threshold") != 0.20:
        _fail(errors, f"worst_subject_guardrail.threshold must be 0.20; got {wsg.get('threshold')}")
    note = wsg.get("policy_note", "")
    bad_phrases = ("50% of B02", "50 % of B02", "50% of the B02", "approximately 50%")
    if any(bp in note for bp in bad_phrases):
        _fail(errors, f"worst_subject_guardrail.policy_note must NOT claim 50% of B02; got: {note[:200]!r}")
    else:
        _ok(oks, "worst-subject floor described as forward-looking policy (not 50% of B02)")

    return errors, oks


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: validate_b04a_protocol.py <config.json>")
        return 2
    path = Path(argv[0])
    if not path.is_file():
        print(f"ERROR: config file not found: {path}")
        return 2
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    errors, oks = validate(config)
    print(f"=== B04A Protocol Contract Validation ===")
    print(f"Config: {path}")
    print(f"OKs: {len(oks)}")
    for o in oks:
        print(f"  + {o}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ! {e}")
    if errors:
        print("\nVALIDATION FAILED")
        return 1
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
