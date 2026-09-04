"""Tests for the B11F AutoDL no-training preflight preparation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts/validate_slp8_b11f_autodl_preflight_preparation.py"
MANIFEST = ROOT / "configs/experiments/slp8_b11f_autodl_no_training_preflight_v0.1.json"


def _module():
    spec = importlib.util.spec_from_file_location("b11f_preflight_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _validate_mutation(tmp_path: Path, mutation) -> list[str]:
    payload = deepcopy(_payload())
    mutation(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _module().validate(path, ROOT)


def test_current_preflight_preparation_is_valid():
    assert _module().validate(MANIFEST, ROOT) == []


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda p: p.__setitem__("execution_authorized", True), "execution_authorized"),
        (lambda p: p.__setitem__("autodl_connection_authorized", True), "autodl_connection_authorized"),
        (lambda p: p["preflight"].__setitem__("formal_experiment_id", "reserved"), "EXP-ID"),
        (lambda p: p["preflight"].__setitem__("training_may_run", True), "training"),
        (lambda p: p["preflight"].__setitem__("training_data_may_be_loaded", True), "training"),
        (lambda p: p["test_gate"].__setitem__("test_rows", 1), "TEST"),
        (lambda p: p["bundle"].__setitem__("sha256", "0" * 64), "bundle SHA-256 mismatch"),
        (lambda p: p["preflight"].__setitem__("script_sha256", "0" * 64), "script SHA-256 mismatch"),
        (lambda p: p["preflight"].__setitem__("remote_script_path", "/tmp/drift"), "remote preflight script path"),
        (lambda p: p["preflight"].__setitem__("checkout_path", "/tmp/drift"), "fixed remote path"),
        (lambda p: p["inputs"]["candidate"].__setitem__("sha256", "0" * 64), "candidate Git blob SHA-256 mismatch"),
    ],
)
def test_manifest_drift_fails_closed(tmp_path: Path, mutation, expected: str):
    errors = _validate_mutation(tmp_path, mutation)
    assert any(expected in error for error in errors)


def test_preflight_script_has_no_training_or_formal_run_flags():
    payload = _payload()
    source = (ROOT / payload["preflight"]["script"]).read_text(encoding="utf-8")
    for token in _module().FORBIDDEN_SCRIPT_TOKENS:
        assert token not in source
    assert "EXP-SLP-B11F" not in source
    assert "--validate-only" in source
    assert "--environment-preflight" in source
    assert "TRAINING_NOT_STARTED" in source
    assert "TEST=0" in source


def test_script_hash_accepts_only_crlf_to_lf_checkout_normalization(tmp_path: Path):
    payload = _payload()
    source = (ROOT / payload["preflight"]["script"]).read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "preflight.sh"
    crlf.write_bytes(source.replace(b"\n", b"\r\n"))
    assert _module()._sha256_lf_normalized(crlf) == payload["preflight"]["script_sha256"]

    crlf.write_bytes(crlf.read_bytes() + b"# drift\r\n")
    assert _module()._sha256_lf_normalized(crlf) != payload["preflight"]["script_sha256"]
