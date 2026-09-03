"""Contract tests for the B11F CPU-only production-wiring smoke."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_slp8_b11f_production_wiring.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("b11f_wiring_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_parses_and_has_no_test_loader_or_formal_run_entrypoint():
    source = _source()
    ast.parse(source)
    assert "load_test=True" not in source
    assert "run_final_fit" not in source
    assert "train_one_unit" not in source
    assert "torch.save" not in source
    assert "EXP-SLP-B11F" not in source


def test_smoke_forces_cpu_before_torch_import_and_exact_numpy_conversion():
    source = _source()
    hide = source.index('os.environ["CUDA_VISIBLE_DEVICES"] = "-1"')
    torch_import = source.index("import torch", hide)
    assert hide < torch_import
    assert 'torch.from_numpy(weights_numpy).to("cpu").to(torch.float32)' in source
    assert 'build_model(MODEL, "cpu")' in source
    assert '"device": "cpu"' in source
    assert '"gpu_training_run": False' in source
    assert '"autodl_connected": False' in source


def test_smoke_uses_real_production_chain_and_one_optimizer_step():
    source = _source()
    for symbol in (
        "load_protocol",
        "load_development_samples",
        "compute_fold_normalization_from_samples",
        "compute_fold_class_weights_from_samples",
        "class_weights_to_tensor",
        "Slp8RegionDataset",
        "build_dataloader",
        "build_model",
        "deterministic_cross_entropy_2d",
    ):
        assert symbol in source
    assert source.count("optimizer.step()") == 1
    assert "for epoch" not in source
    assert "SMOKE_BATCH_SIZE = 1" in source
    assert "STATISTICS_SAMPLE_COUNT = 128" in source
    assert '"production_batch_size": BATCH_SIZE' in source
    assert "DETERMINISTIC_REAL_DEVELOPMENT_SUBSET_NOT_PRODUCTION_ESTIMATE" in source
    assert '"data_manifest_sha256": sha256_file(freeze_manifest)' in source
    assert "manifest_sha256(freeze_manifest)" not in source


def test_smoke_has_zero_test_carriers_and_refuses_output_overwrite():
    source = _source()
    assert '"test_access": False' in source
    assert '"test_rows": 0' in source
    assert '"test_labels": 0' in source
    assert '"test_onehot": 0' in source
    assert "refusing to overwrite existing smoke output" in source
    assert "allow_nan=False" in source


def test_atomic_output_is_valid_json_and_cannot_be_overwritten(tmp_path: Path):
    module = _load_script_module()
    output = tmp_path / "summary.json"
    module._atomic_write_new(output, {"result": "PASS", "test_rows": 0})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "result": "PASS",
        "test_rows": 0,
    }
    with pytest.raises(module.SmokeError, match="refusing to overwrite"):
        module._atomic_write_new(output, {"result": "CHANGED"})


def test_atomic_output_does_not_clobber_competing_target(tmp_path: Path, monkeypatch):
    module = _load_script_module()
    output = tmp_path / "summary.json"
    original_link = module.os.link

    def competing_link(source, target):
        Path(target).write_text("SENTINEL", encoding="utf-8")
        return original_link(source, target)

    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(module.SmokeError, match="refusing to overwrite"):
        module._atomic_write_new(output, {"result": "PASS"})
    assert output.read_text(encoding="utf-8") == "SENTINEL"
    assert set(tmp_path.iterdir()) == {output}


def test_atomic_publish_does_not_use_replacing_rename():
    source = _source()
    assert "os.replace" not in source
    assert "os.link(temporary, path)" in source
