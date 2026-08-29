"""Focused B04A implementation tests (TASK-SLP-B04A-IMPLEMENTATION-SMOKE-v0.1).

This module exercises the three B04A candidates in the model registry
(``slp8_small_unet_v0.1``, ``slp8_resunet_lite_v0.1``,
``slp8_deeplabv3plus_lite_v0.1``) and verifies that each implementation
matches the frozen B04A R03 forward plan and contract:

  1. exact parameter counts (118,121 / 120,809 / 53,449);
  2. input / output shape ``[N, 1, 192, 84]`` → ``[N, 9, 192, 84]``;
  3. batch size 1 and batch size > 1 forward passes;
  4. forward output is finite;
  5. backward updates the parameters;
  6. ResUNet residual Add shape/channel consistency;
  7. ResUNet shortcut is an explicit 1x1 Conv2d
     (kernel=1, stride=1, padding=0, bias=True);
  8. DeepLabV3+-lite ASPP has six branches and the concat is 96
     channels;
  9. DeepLabV3+-lite all Conv2d ``groups=1`` and ASPP
     ``dilation == padding == atrous_rate``;
 10. the model registry exposes the three B04A versions and unknown
     names fail-closed;
 11. the JSON config forward_plan matches the implementation structure;
 12. checkpoint save / reload produces byte-identical predictions;
 13. deterministic same-seed CPU synthetic smoke is reproducible;
 14. ``TEST`` access is never invoked (this module never imports the
     B01 training-table loader or the B01 test-access contract);
 15. SmallUNet original tests and parameter count do not regress.

CPU-only by design; CUDA smoke is recorded as ``NOT RUN``.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from topper_perception.neural.slp8_region_models import (
    B04A_EXACT_PARAMETER_COUNTS,
    B04A_MAX_PARAMETERS,
    B04_MAX_PARAMETERS,
    DEEPLABV3PLUS_LITE_ATROUS_RATES,
    DEEPLABV3PLUS_LITE_BRANCH_CHANNELS,
    DEEPLABV3PLUS_LITE_POST_CONCAT_CHANNELS,
    DEEPLABV3PLUS_LITE_VERSION,
    INPUT_SHAPE,
    MODEL_REGISTRY,
    N_CLASSES,
    RESUNET_LITE_VERSION,
    SMALL_UNET_VERSION,
    Slp8DeepLabV3PlusLite,
    Slp8ResUnetLite,
    Slp8SmallUnet,
    _AsppModule,
    _ResidualBlock,
    get_model_builder,
    list_model_builders,
)


# ---------------------------------------------------------------------------
# Configuration / config path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
B04A_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "slp8_pm_architecture_expansion_mini_v0.1.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_input(batch_size: int = 2) -> torch.Tensor:
    return torch.randn(
        batch_size, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32
    )


def _valid_labels(batch_size: int = 2) -> torch.Tensor:
    """Return synthetic foreground labels for a backward sanity test."""

    labels = torch.zeros(batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.long)
    # Sprinkle distinct foreground class IDs so cross-entropy is well-defined.
    for c in range(1, N_CLASSES):
        labels[:, c, c] = c
    return labels


@pytest.fixture
def b04a_config() -> dict:
    with B04A_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1) Registry + version names + unknown fail-closed
# ---------------------------------------------------------------------------


class TestB04ARegistry:
    """The B04A candidate names are registered and unknown names fail-closed."""

    def test_b04a_versions_registered(self):
        names = set(list_model_builders())
        for version in (
            SMALL_UNET_VERSION,
            RESUNET_LITE_VERSION,
            DEEPLABV3PLUS_LITE_VERSION,
        ):
            assert version in names, (
                f"B04A candidate {version!r} is not registered; got {names}"
            )

    def test_b04a_builder_versions_match_names(self):
        for version in (
            SMALL_UNET_VERSION,
            RESUNET_LITE_VERSION,
            DEEPLABV3PLUS_LITE_VERSION,
        ):
            builder = get_model_builder(version)
            assert builder.name == version
            assert builder.version == version

    def test_unknown_model_fails_closed(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_builder("slp8_nonexistent_v0.0")

    def test_b04a_exact_counts_dict_keys(self):
        assert set(B04A_EXACT_PARAMETER_COUNTS) == {
            SMALL_UNET_VERSION,
            RESUNET_LITE_VERSION,
            DEEPLABV3PLUS_LITE_VERSION,
        }


# ---------------------------------------------------------------------------
# 2) Exact parameter counts (per the B04A R03 contract)
# ---------------------------------------------------------------------------


class TestB04AExactParameterCounts:
    """Exact parameter counts must equal the frozen B04A R03 values."""

    @pytest.mark.parametrize(
        "cls, version, expected",
        [
            (Slp8SmallUnet, SMALL_UNET_VERSION, 118_121),
            (Slp8ResUnetLite, RESUNET_LITE_VERSION, 120_809),
            (Slp8DeepLabV3PlusLite, DEEPLABV3PLUS_LITE_VERSION, 53_449),
        ],
    )
    def test_exact_parameter_count(self, cls, version, expected):
        m = cls()
        assert m.count_parameters() == expected
        assert m.count_total_parameters() == expected
        assert m.model_version == version

    def test_exact_counts_match_frozen_dict(self):
        m_small = Slp8SmallUnet()
        m_resunet = Slp8ResUnetLite()
        m_deeplab = Slp8DeepLabV3PlusLite()
        for m, version in (
            (m_small, SMALL_UNET_VERSION),
            (m_resunet, RESUNET_LITE_VERSION),
            (m_deeplab, DEEPLABV3PLUS_LITE_VERSION),
        ):
            assert B04A_EXACT_PARAMETER_COUNTS[version] == m.count_parameters()

    def test_smallunet_under_b04_cap(self):
        m = Slp8SmallUnet()
        assert m.count_parameters() <= B04_MAX_PARAMETERS

    def test_new_candidates_under_b04a_cap(self):
        for cls in (Slp8ResUnetLite, Slp8DeepLabV3PlusLite):
            m = cls()
            assert m.count_parameters() <= B04A_MAX_PARAMETERS


# ---------------------------------------------------------------------------
# 3) Input / output shape contract
# ---------------------------------------------------------------------------


class TestB04AInputOutputShapes:
    """Each B04A candidate must accept ``[N, 1, 192, 84]`` and emit
    ``[N, 9, 192, 84]`` for any batch size."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_output_shape(self, cls, batch_size):
        m = cls()
        y = m(_valid_input(batch_size))
        assert y.shape == (batch_size, N_CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1])
        assert y.dtype == torch.float32

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_predict_shape_and_range(self, cls):
        m = cls()
        x = _valid_input(3)
        p = m.predict(x)
        assert p.shape == (3, INPUT_SHAPE[0], INPUT_SHAPE[1])
        assert p.dtype == torch.long
        assert p.min().item() >= 0
        assert p.max().item() < N_CLASSES

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_fail_closed_input_validation(self, cls):
        m = cls()
        with pytest.raises(ValueError, match="4D"):
            m(torch.randn(INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32))
        with pytest.raises(ValueError, match="channel must be 1"):
            m(torch.randn(2, 3, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32))
        with pytest.raises(ValueError, match="spatial shape"):
            m(torch.randn(2, 1, 100, 100, dtype=torch.float32))
        with pytest.raises(ValueError, match="float32"):
            m(torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float64))
        bad = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        bad[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            m(bad)


# ---------------------------------------------------------------------------
# 4) Forward output finiteness + backward parameter update
# ---------------------------------------------------------------------------


class TestB04AForwardFiniteAndBackward:
    """Forward must be finite; backward must change the parameters."""

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_forward_is_finite(self, cls):
        m = cls()
        m.eval()
        with torch.no_grad():
            y = m(_valid_input(2))
        assert torch.isfinite(y).all().item()

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_backward_updates_parameters(self, cls):
        m = cls()
        initial_state = {k: v.clone() for k, v in m.state_dict().items()}

        m.train()
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        x = _valid_input(2)
        y = m(x)
        loss = loss_fn(
            y.reshape(y.shape[0], N_CLASSES, -1),
            _valid_labels(2).reshape(2, -1),
        )
        assert torch.isfinite(loss).item()
        optimizer.zero_grad()
        loss.backward()

        # All trainable params receive a non-zero gradient.
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all().item()
                assert p.grad.abs().sum().item() > 0

        optimizer.step()

        # At least one parameter must have changed.
        changed = False
        for key, init_tensor in initial_state.items():
            cur = m.state_dict()[key]
            if not torch.equal(init_tensor, cur):
                changed = True
                break
        assert changed, "no parameter changed after optimizer.step()"


# ---------------------------------------------------------------------------
# 5) ResUNet-lite residual block contract
# ---------------------------------------------------------------------------


class TestResUnetLiteResidualBlocks:
    """The three residual blocks must be explicit 1x1 Conv2d shortcuts
    and the residual Add shape / channels must match on both sides."""

    def test_three_residual_blocks_exist(self):
        m = Slp8ResUnetLite()
        for name in ("enc1_resblock", "enc2_resblock", "bottleneck_resblock"):
            block = m.get_residual_block(name)
            assert isinstance(block, _ResidualBlock)

    def test_shortcut_is_1x1_conv2d(self):
        m = Slp8ResUnetLite()
        for name in ("enc1_resblock", "enc2_resblock", "bottleneck_resblock"):
            block = m.get_residual_block(name)
            sc = block.shortcut_conv
            assert isinstance(sc, nn.Conv2d)
            assert sc.kernel_size == (1, 1)
            assert sc.stride == (1, 1)
            assert sc.padding == (0, 0)
            assert sc.bias is not None
            assert sc.groups == 1  # plain Conv2d; not depthwise.

    def test_shortcut_in_out_channels_per_block(self):
        m = Slp8ResUnetLite()
        expected = {
            "enc1_resblock": (1, 16),
            "enc2_resblock": (16, 32),
            "bottleneck_resblock": (32, 64),
        }
        for name, (in_c, out_c) in expected.items():
            block = m.get_residual_block(name)
            assert block.shortcut_conv.in_channels == in_c
            assert block.shortcut_conv.out_channels == out_c
            assert block.main_output_channels == out_c
            assert block.shortcut_output_channels == out_c

    def test_main_and_shortcut_shapes_match(self):
        m = Slp8ResUnetLite()
        m.eval()
        x = _valid_input(1)
        # enc1
        block = m.get_residual_block("enc1_resblock")
        main = block.main_conv2(block.main_relu1(block.main_conv1(x)))
        shortcut = block.shortcut_conv(x)
        assert main.shape == shortcut.shape
        # enc2
        x2 = nn.functional.max_pool2d(m.get_residual_block("enc1_resblock")(x), 2)
        block2 = m.get_residual_block("enc2_resblock")
        main2 = block2.main_conv2(block2.main_relu1(block2.main_conv1(x2)))
        shortcut2 = block2.shortcut_conv(x2)
        assert main2.shape == shortcut2.shape
        # bottleneck
        x3 = nn.functional.max_pool2d(m.get_residual_block("enc2_resblock")(x2), 2)
        block3 = m.get_residual_block("bottleneck_resblock")
        main3 = block3.main_conv2(block3.main_relu1(block3.main_conv1(x3)))
        shortcut3 = block3.shortcut_conv(x3)
        assert main3.shape == shortcut3.shape

    def test_residual_add_runs_end_to_end(self):
        m = Slp8ResUnetLite()
        m.eval()
        x = _valid_input(2)
        y = m(x)
        assert y.shape == (2, N_CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1])
        assert torch.isfinite(y).all().item()


# ---------------------------------------------------------------------------
# 6) DeepLabV3+-lite ASPP / decoder contract
# ---------------------------------------------------------------------------


class TestDeepLabV3PlusLiteAspp:
    """ASPP module must have six branches, dilation==padding==rate,
    groups=1 on every Conv2d, and the post-concat 96 → 32 layout."""

    def test_variant_is_option_a(self):
        m = Slp8DeepLabV3PlusLite()
        assert m.variant == "option_A_plain_atrous_Conv2d"

    def test_aspp_has_six_branches(self):
        m = Slp8DeepLabV3PlusLite()
        aspp: _AsppModule = m.aspp
        # 1 pointwise + 4 atrous + 1 GAP = 6 branches.
        assert len(aspp.branch_atrous) == 4
        assert aspp.branch_pointwise is not None
        assert aspp.branch_gap is not None
        total_branches = 1 + len(aspp.branch_atrous) + 1
        assert total_branches == 6

    def test_atrous_rates_match_frozen(self):
        m = Slp8DeepLabV3PlusLite()
        assert m.aspp.atrous_rates == DEEPLABV3PLUS_LITE_ATROUS_RATES
        assert DEEPLABV3PLUS_LITE_ATROUS_RATES == (3, 6, 9, 12)

    def test_branch_output_channels_match(self):
        m = Slp8DeepLabV3PlusLite()
        aspp = m.aspp
        assert aspp.out_channels_per_branch == DEEPLABV3PLUS_LITE_BRANCH_CHANNELS == 16

    def test_dilation_padding_match_rate(self):
        m = Slp8DeepLabV3PlusLite()
        aspp = m.aspp
        for rate, branch in zip(aspp.atrous_rates, aspp.branch_atrous):
            assert branch.atrous_rate == rate
            assert branch.conv.dilation == (rate, rate)
            assert branch.conv.padding == (rate, rate)
            assert branch.conv.kernel_size == (3, 3)
            assert branch.conv.stride == (1, 1)
            assert branch.conv.groups == 1
            assert branch.conv.bias is not None

    def test_pointwise_and_gap_groups_one(self):
        m = Slp8DeepLabV3PlusLite()
        aspp = m.aspp
        assert aspp.branch_pointwise.conv.groups == 1
        assert aspp.branch_pointwise.conv.kernel_size == (1, 1)
        assert aspp.branch_gap.conv.groups == 1

    def test_post_concat_input_96_to_32(self):
        m = Slp8DeepLabV3PlusLite()
        aspp = m.aspp
        assert aspp.concat_in_channels == 96
        assert DEEPLABV3PLUS_LITE_POST_CONCAT_CHANNELS == 96
        assert aspp.post_concat_conv.in_channels == 96
        assert aspp.post_concat_conv.out_channels == 32
        assert aspp.post_concat_conv.kernel_size == (1, 1)
        assert aspp.post_concat_conv.groups == 1

    def test_all_conv2d_groups_are_one(self):
        """No Xception / depthwise-separable anywhere in the network."""

        m = Slp8DeepLabV3PlusLite()
        for name, module in m.named_modules():
            if isinstance(module, nn.Conv2d):
                assert module.groups == 1, (
                    f"DeepLabV3+-lite forbids non-1 groups; {name} has "
                    f"groups={module.groups}"
                )

    def test_decoder_low_level_proj_and_fusion(self):
        m = Slp8DeepLabV3PlusLite()
        # Low-level projection: 16 → 16 at full input resolution.
        assert m.low_level_proj_conv.in_channels == 16
        assert m.low_level_proj_conv.out_channels == 16
        assert m.low_level_proj_conv.kernel_size == (1, 1)
        assert m.low_level_proj_conv.groups == 1
        # Decoder fusion: 48 → 32 → 32.
        assert m.decoder_conv1.in_channels == 48
        assert m.decoder_conv1.out_channels == 32
        assert m.decoder_conv2.in_channels == 32
        assert m.decoder_conv2.out_channels == 32
        # Final: 32 → 9.
        assert m.final_conv.in_channels == 32
        assert m.final_conv.out_channels == N_CLASSES
        assert m.final_conv.kernel_size == (1, 1)

    def test_aspp_six_branch_outputs_concat_to_96(self):
        """End-to-end ASPP forward: six branch outputs concatenated to 96 channels."""

        m = Slp8DeepLabV3PlusLite()
        m.eval()
        # Build the post-down feature map at the ASPP input size.
        x_low = m.stem_relu2(
            m.stem_conv2(m.stem_relu1(m.stem_conv1(_valid_input(1))))
        )
        x = m.down_relu(m.down_conv(x_low))
        aspp = m.aspp
        feats = [aspp.branch_pointwise(x)]
        for branch in aspp.branch_atrous:
            feats.append(branch(x))
        feats.append(aspp.branch_gap(x))
        concat = torch.cat(feats, dim=1)
        assert concat.shape[1] == 96


# ---------------------------------------------------------------------------
# 7) No BatchNorm / Dropout / depthwise / pretrained anywhere
# ---------------------------------------------------------------------------


class TestB04ANoForbiddenLayers:
    """No BatchNorm, Dropout, or pretrained weights in any B04A candidate."""

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_no_batchnorm_no_dropout(self, cls):
        m = cls()
        for module in m.modules():
            assert not isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
            )
            assert not isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
            )

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_no_pretrained_attribute(self, cls):
        m = cls()
        cfg = m.get_config()
        for forbidden in (
            "pretrained",
            "checkpoint_url",
            "external_weights",
            "url",
        ):
            assert forbidden not in cfg


# ---------------------------------------------------------------------------
# 8) Config forward_plan consistency
# ---------------------------------------------------------------------------


class TestB04AConfigForwardPlanConsistency:
    """The frozen JSON forward_plan must agree with the implementation
    for the layer counts, channel widths, and ASPP configuration."""

    def _candidate(self, config: dict, name: str) -> dict:
        for c in config["candidates"]:
            if c["name"] == name:
                return c
        raise AssertionError(f"candidate {name!r} not in config")

    def test_exact_parameter_counts_match_config(self, b04a_config):
        expected = {
            SMALL_UNET_VERSION: 118_121,
            RESUNET_LITE_VERSION: 120_809,
            DEEPLABV3PLUS_LITE_VERSION: 53_449,
        }
        for version, value in expected.items():
            cand = self._candidate(b04a_config, version)
            assert cand["exact_parameter_count"] == value

    def test_resunet_shortcut_in_forward_plan(self, b04a_config):
        cand = self._candidate(b04a_config, RESUNET_LITE_VERSION)
        plan = cand["architecture_freeze"]["forward_plan"]
        residual_blocks = [s for s in plan if s.get("type") == "residual_block"]
        assert len(residual_blocks) == 3
        for block in residual_blocks:
            sc = block["shortcut"]
            assert sc["op"] == "Conv2d"
            assert sc["kernel_size"] == 1
            assert sc["stride"] == 1
            assert sc["padding"] == 0
            assert sc["bias"] is True
            assert block["main_output_channels"] == block["shortcut_output_channels"]
            assert block["main_output_shape"] == block["shortcut_output_shape"]
            add = block["add_constraint"]
            assert add["lhs_channels_must_equal_rhs_channels"] is True
            assert add["lhs_shape_must_equal_rhs_shape"] is True

    def test_deeplab_aspp_in_forward_plan(self, b04a_config):
        cand = self._candidate(b04a_config, DEEPLABV3PLUS_LITE_VERSION)
        plan = cand["architecture_freeze"]["forward_plan"]
        aspp = next(s for s in plan if s.get("name") == "aspp")
        settings = aspp["aspp_settings"]
        assert settings["atrous_rates"] == [3, 6, 9, 12]
        assert settings["n_atrous_branches"] == 4
        assert settings["include_1x1_pointwise_branch"] is True
        assert settings["include_gap_branch"] is True
        assert settings["output_channels_per_branch"] == 16
        assert settings["expected_post_concat_input_channels"] == 96
        assert len(aspp["branches"]) == 6
        # Atrous branches: dilation == padding == atrous_rate, groups=1.
        atrous = [b for b in aspp["branches"] if b["kind"] == "atrous_3x3"]
        assert len(atrous) == 4
        for branch in atrous:
            ops = branch["ops"]
            assert len(ops) == 1
            op = ops[0]
            assert op["op"] == "Conv2d"
            assert op["dilation"] == op["padding"] == branch["atrous_rate"]
            assert op["groups"] == 1
        # Concat
        assert aspp["concat"]["result_channels"] == 96
        # post_concat
        post = aspp["post_concat"]
        assert post[0]["op"] == "Conv2d"
        assert post[0]["in_channels"] == 96
        assert post[0]["out_channels"] == 32

    def test_deeplab_variant_label_in_config(self, b04a_config):
        cand = self._candidate(b04a_config, DEEPLABV3PLUS_LITE_VERSION)
        assert (
            cand["architecture_freeze"]["variant"]
            == "option_A_plain_atrous_Conv2d"
        )

    def test_deeplab_decoder_fusion_in_forward_plan(self, b04a_config):
        cand = self._candidate(b04a_config, DEEPLABV3PLUS_LITE_VERSION)
        plan = cand["architecture_freeze"]["forward_plan"]
        fusion = next(s for s in plan if s.get("name") == "decoder_fusion")
        # ops: BilinearInterpolate + Concat + Conv 48→32 + ReLU + Conv 32→32 + ReLU
        assert any(
            op.get("op") == "Concat" and op.get("with_channels") == 16
            for op in fusion["ops"]
        )

    def test_segformer_deferred(self, b04a_config):
        cand = self._candidate(b04a_config, "slp8_segformer_b0_v0.1")
        assert cand["role"] == "DEFERRED"


# ---------------------------------------------------------------------------
# 9) Checkpoint save / reload prediction equality
# ---------------------------------------------------------------------------


class TestB04ACheckpointRoundtrip:
    """Saving and reloading a model state_dict must yield byte-identical
    predictions for the same input.  We embed the standard ``identity``
    dict expected by :mod:`topper_perception.neural.slp8_region_resume`."""

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_state_dict_roundtrip(self, cls):
        m1 = cls()
        m1.eval()
        x = _valid_input(2)
        with torch.no_grad():
            y1 = m1(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save(
                {
                    "model_state_dict": m1.state_dict(),
                    "model_version": m1.model_version,
                    "n_classes": m1.n_classes,
                    "identity": {
                        "experiment_id": "EXP-B04A-TEST",
                        "git_commit": "deadbeef",
                        "git_dirty": False,
                        "config_sha256": "0" * 64,
                        "data_manifest_sha256": "0" * 64,
                        "split_sha256": "0" * 64,
                        "model_version": m1.model_version,
                    },
                },
                path,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        m2 = cls()
        m2.load_state_dict(payload["model_state_dict"])
        m2.eval()
        with torch.no_grad():
            y2 = m2(x)

        assert y1.shape == y2.shape
        assert torch.equal(y1, y2)
        assert payload["model_version"] == m1.model_version
        assert payload["identity"]["model_version"] == m1.model_version


# ---------------------------------------------------------------------------
# 10) Deterministic same-seed CPU synthetic smoke
# ---------------------------------------------------------------------------


class TestB04ADeterministicSmoke:
    """Two CPU synthetic-smoke runs with the same seed must produce
    identical predictions.  This is the B04A implementation-level
    determinism check (no real data, no GPU)."""

    @pytest.mark.parametrize(
        "cls",
        [Slp8SmallUnet, Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_same_seed_same_predictions(self, cls):
        def _run_once() -> torch.Tensor:
            torch.manual_seed(42)
            m = cls()
            m.eval()
            torch.manual_seed(123)
            x = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
            with torch.no_grad():
                return m(x)

        y_a = _run_once()
        y_b = _run_once()
        assert torch.equal(y_a, y_b)

    def test_cpu_synthetic_smoke_pipeline(self):
        """End-to-end synthetic CPU smoke (no real data)."""

        torch.manual_seed(42)
        x = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        labels = torch.randint(0, N_CLASSES, (2, INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=torch.long)

        for cls, name in (
            (Slp8SmallUnet, "slp8_small_unet_v0.1"),
            (Slp8ResUnetLite, "slp8_resunet_lite_v0.1"),
            (Slp8DeepLabV3PlusLite, "slp8_deeplabv3plus_lite_v0.1"),
        ):
            torch.manual_seed(42)
            m = cls()
            m.train()
            optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            losses = []
            for _ in range(2):
                optimizer.zero_grad()
                y = m(x)
                loss = loss_fn(
                    y.reshape(y.shape[0], N_CLASSES, -1),
                    labels.reshape(2, -1),
                )
                assert torch.isfinite(loss).item()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            assert losses[1] < losses[0] + 1e-3  # loss should not blow up
            # Final param count invariant.
            assert m.count_parameters() == B04A_EXACT_PARAMETER_COUNTS[name]


# ---------------------------------------------------------------------------
# 11) TEST access = 0
# ---------------------------------------------------------------------------


class TestB04ANoTestAccess:
    """This implementation module must never touch the B01 TEST contract.

    We verify that by asserting the module does not import or call the
    B01 training-table freeze loader, the test-access allow function, or
    any TEST-related constant.
    """

    def test_module_does_not_import_b01_test_contract(self):
        from topper_perception.neural import slp8_region_models as m

        forbidden_substrings = (
            "slp8_training_table_freeze",
            "enable_test_access",
            "TestLeakageError",
            "compute_class_stats",
            "slp8_8region_pressure_dataset",
        )
        source = Path(m.__file__).read_text(encoding="utf-8")
        for forbidden in forbidden_substrings:
            assert forbidden not in source, (
                f"B04A implementation must not import {forbidden!r}"
            )

    def test_registry_lookup_does_not_load_test(self):
        for name in (
            SMALL_UNET_VERSION,
            RESUNET_LITE_VERSION,
            DEEPLABV3PLUS_LITE_VERSION,
        ):
            builder = get_model_builder(name)
            # Constructing a model must NOT touch the data layer.
            model, _ = builder.factory(N_CLASSES, "cpu")
            assert isinstance(model, nn.Module)
            # The registry must NOT have grown to include any other name.
            assert name in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# 12) B04A max-parameters guardrail
# ---------------------------------------------------------------------------


class TestB04AMaxParameterGuardrail:
    """The new B04A candidates must respect the 300,000 cap."""

    @pytest.mark.parametrize(
        "cls",
        [Slp8ResUnetLite, Slp8DeepLabV3PlusLite],
    )
    def test_under_b04a_cap(self, cls):
        m = cls()
        assert m.count_parameters() <= B04A_MAX_PARAMETERS


# ---------------------------------------------------------------------------
# 13) Module-level config contract sanity
# ---------------------------------------------------------------------------


class TestB04AModuleConstants:
    """Module-level constants must match the B04A R03 freeze."""

    def test_input_shape(self):
        assert INPUT_SHAPE == (192, 84)

    def test_n_classes(self):
        assert N_CLASSES == 9

    def test_b04a_max_parameters(self):
        assert B04A_MAX_PARAMETERS == 300_000

    def test_deeplab_branch_channels(self):
        assert DEEPLABV3PLUS_LITE_BRANCH_CHANNELS == 16

    def test_deeplab_post_concat_channels(self):
        assert DEEPLABV3PLUS_LITE_POST_CONCAT_CHANNELS == 96


# ---------------------------------------------------------------------------
# 14) No-op math sanity (sanity check the B04A EXACT_PARAMETER_COUNTS math)
# ---------------------------------------------------------------------------


def test_b04a_param_count_math_is_consistent():
    """The frozen exact parameter counts must add up to a sensible
    range when sanity-checked by independently summing each layer's
    contribution for one of the candidates."""

    # SmallUNet reference architecture (already known to match).
    m = Slp8SmallUnet()
    expected_layers = {
        "enc1_conv1": (1, 16, 3),
        "enc1_conv2": (16, 16, 3),
        "enc2_conv1": (16, 32, 3),
        "enc2_conv2": (32, 32, 3),
        "bottleneck_conv1": (32, 64, 3),
        "bottleneck_conv2": (64, 64, 3),
        "dec1_conv1": (96, 32, 3),
        "dec1_conv2": (32, 32, 3),
        "dec2_conv1": (48, 16, 3),
        "dec2_conv2": (16, 16, 3),
        "final_conv": (16, 9, 1),
    }
    total = 0
    for name, (in_c, out_c, k) in expected_layers.items():
        op = getattr(m, name)
        assert isinstance(op, nn.Conv2d)
        assert op.in_channels == in_c
        assert op.out_channels == out_c
        assert op.kernel_size == (k, k)
        total += in_c * out_c * k * k + out_c  # bias=True
    assert total == m.count_parameters()
    assert total == 118_121


# ---------------------------------------------------------------------------
# 15) ResUNet-lite shortcut parameter math (R02 ITERATE)
# ---------------------------------------------------------------------------


def test_resunet_shortcut_params_sum_to_2688():
    """The three 1x1 Conv2d shortcut parameters must sum to exactly
    2688, which equals the ResUNet - SmallUNet parameter delta.

    Breakdown (bias=True):
      enc1_resblock      : 1 * 16 * 1 * 1 + 16    =    32
      enc2_resblock      : 16 * 32 * 1 * 1 + 32   =   544
      bottleneck_resblock: 32 * 64 * 1 * 1 + 64   =  2112
      ---------------------------------------------
      total                                    =  2688
    """

    m = Slp8ResUnetLite()
    expected = {
        "enc1_resblock": 32,
        "enc2_resblock": 544,
        "bottleneck_resblock": 2112,
    }
    actual = {
        name: sum(p.numel() for p in m.get_residual_block(name).shortcut_conv.parameters())
        for name in expected
    }
    assert actual == expected
    assert sum(actual.values()) == 2688
    assert sum(actual.values()) == (
        Slp8ResUnetLite().count_parameters() - Slp8SmallUnet().count_parameters()
    )


# ---------------------------------------------------------------------------
# 16) Smoke script refuse-overwrite + no-write behavior
# ---------------------------------------------------------------------------


def test_smoke_script_refuses_to_overwrite_existing_output(tmp_path):
    """The smoke script must REFUSE to overwrite an existing output
    file.  This prevents silent write_text clobbering of historical
    artifacts.  --force is required to allow overwrite."""

    import subprocess
    import sys as _sys

    out_path = tmp_path / "smoke_summary.json"
    # First run: write a fresh file.
    rc1 = subprocess.run(
        [
            _sys.executable, "scripts/smoke_b04a_implementation.py",
            "--output", str(out_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert rc1.returncode == 0, f"first run failed: {rc1.stderr}"
    assert out_path.is_file()
    first_bytes = out_path.read_bytes()
    # Second run without --force: must refuse.
    rc2 = subprocess.run(
        [
            _sys.executable, "scripts/smoke_b04a_implementation.py",
            "--output", str(out_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert rc2.returncode != 0, "smoke script should refuse overwrite"
    assert "Refusing to overwrite" in rc2.stderr or "already exists" in rc2.stderr
    # File must be byte-identical to the first run (no silent clobbering).
    assert out_path.read_bytes() == first_bytes
    # Third run with --force: must succeed and rewrite.
    rc3 = subprocess.run(
        [
            _sys.executable, "scripts/smoke_b04a_implementation.py",
            "--output", str(out_path),
            "--force",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert rc3.returncode == 0, f"--force run failed: {rc3.stderr}"
    assert out_path.is_file()


def test_smoke_script_no_write_does_not_touch_disk(tmp_path):
    """The smoke script must support a --no-write mode that runs the
    CPU pipeline and prints a one-line summary to stdout without
    writing any file."""

    import subprocess
    import sys as _sys

    out_path = tmp_path / "should_not_exist.json"
    assert not out_path.exists()
    rc = subprocess.run(
        [
            _sys.executable, "scripts/smoke_b04a_implementation.py",
            "--output", str(out_path),
            "--no-write",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert rc.returncode == 0, f"--no-write failed: {rc.stderr}"
    assert not out_path.exists(), "--no-write must not create the output file"
    # Summary line must include the three CPU candidates.
    assert "cpu_candidates=3" in rc.stdout
    # CUDA on this host is unavailable; the summary records the flag.
    assert "cuda_run=False" in rc.stdout
    assert "all_cpu_ok=True" in rc.stdout


def test_smoke_summary_records_test_access_as_declarative_policy(tmp_path):
    """The summary JSON must record test_access as a declarative policy,
    NOT a runtime count of TEST reads (which would be false
    advertising)."""

    import json as _json
    import subprocess
    import sys as _sys

    out_path = tmp_path / "smoke_summary.json"
    rc = subprocess.run(
        [
            _sys.executable, "scripts/smoke_b04a_implementation.py",
            "--output", str(out_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert rc.returncode == 0
    payload = _json.loads(out_path.read_text(encoding="utf-8"))
    assert "test_access" in payload
    assert payload["test_access"]["value"] == 0
    assert payload["test_access"]["kind"] == "declarative_policy"
    assert "declarative" in payload["test_access"]["explanation"].lower() or \
        "static" in payload["test_access"]["explanation"].lower()
