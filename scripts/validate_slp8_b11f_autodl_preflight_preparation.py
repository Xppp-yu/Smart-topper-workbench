"""Validate the B11F AutoDL no-training preflight preparation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/experiments/slp8_b11f_autodl_no_training_preflight_v0.1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_SCRIPT_TOKENS = (
    "--run-authorized",
    "--resume-authorized",
    "--experiment-id",
    "--output-dir",
    "optimizer.step",
    "loss.backward",
    "torch.save",
    "load_test=True",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf_normalized(path: Path) -> str:
    """Hash a shell-script transfer payload independent of Windows checkout EOLs."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_blob_sha256(repo_root: Path, revision: str, path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo_root,
    )
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preflight manifest must be a JSON object")
    return payload


def validate(manifest_path: Path, repo_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:
        return [f"manifest unreadable: {exc}"]

    if payload.get("status") != "PREPARATION_ONLY_AUTODL_NOT_AUTHORIZED":
        errors.append("status must remain preparation-only and AutoDL-not-authorized")
    if payload.get("execution_authorized") is not False:
        errors.append("execution_authorized must be strict false")
    if payload.get("autodl_connection_authorized") is not False:
        errors.append("autodl_connection_authorized must be strict false")

    runner = payload.get("runner", {})
    runner_sha = runner.get("git_sha")
    if not isinstance(runner_sha, str) or not GIT_SHA_RE.fullmatch(runner_sha):
        errors.append("runner git SHA must be 40 lowercase hex characters")
    else:
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{runner_sha}^{{commit}}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", runner_sha, "origin/main"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            errors.append("runner SHA is missing or is not an origin/main ancestor")
    if runner.get("git_dirty_required") is not False:
        errors.append("runner must require git_dirty=false")
    if runner.get("must_be_origin_main_ancestor") is not True:
        errors.append("runner origin/main ancestry gate missing")

    bundle = payload.get("bundle", {})
    bundle_path = repo_root / str(bundle.get("local_path", ""))
    bundle_sha = bundle.get("sha256")
    if not isinstance(bundle_sha, str) or not SHA256_RE.fullmatch(bundle_sha):
        errors.append("bundle SHA-256 invalid")
    elif not bundle_path.is_file():
        errors.append("bundle file missing")
    else:
        if _sha256(bundle_path) != bundle_sha:
            errors.append("bundle SHA-256 mismatch")
        if bundle_path.stat().st_size != bundle.get("size_bytes"):
            errors.append("bundle size mismatch")
        result = subprocess.run(
            ["git", "bundle", "verify", str(bundle_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or "complete history" not in (result.stdout + result.stderr):
            errors.append("bundle does not verify as complete history")
    if bundle.get("ref") != "refs/heads/main" or bundle.get("complete_history") is not True:
        errors.append("bundle ref/history contract mismatch")

    inputs = payload.get("inputs", {})
    for key in ("config", "candidate"):
        item = inputs.get(key, {})
        tracked_path = str(item.get("path", ""))
        expected = item.get("sha256")
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            errors.append(f"{key} SHA-256 invalid")
        elif isinstance(runner_sha, str) and GIT_SHA_RE.fullmatch(runner_sha):
            try:
                if _git_blob_sha256(repo_root, runner_sha, tracked_path) != expected:
                    errors.append(f"{key} Git blob SHA-256 mismatch")
            except subprocess.CalledProcessError:
                errors.append(f"{key} Git blob missing at runner SHA")

    freeze = inputs.get("b01_freeze_manifest", {})
    freeze_path = repo_root / str(freeze.get("local_path", ""))
    freeze_expected = freeze.get("sha256")
    if not isinstance(freeze_expected, str) or not SHA256_RE.fullmatch(freeze_expected):
        errors.append("b01_freeze_manifest SHA-256 invalid")
    elif not freeze_path.is_file():
        errors.append("b01_freeze_manifest file missing")
    elif _sha256(freeze_path) != freeze_expected:
        errors.append("b01_freeze_manifest SHA-256 mismatch")

    preflight = payload.get("preflight", {})
    required_false = (
        "formal_output_may_be_created",
        "training_may_run",
        "training_data_may_be_loaded",
        "checkpoint_may_be_created",
        "resume_may_run",
    )
    if any(preflight.get(key) is not False for key in required_false):
        errors.append("preflight write/training/data/resume gates must be strict false")
    if preflight.get("formal_experiment_id") is not None:
        errors.append("preflight must not reserve or use a formal EXP-ID")
    if preflight.get("environment_probe_only") is not True:
        errors.append("preflight must remain an environment probe only")
    remote_script_path = preflight.get("remote_script_path")
    if (
        not isinstance(remote_script_path, str)
        or remote_script_path != "/root/autodl-tmp/preflight_slp8_b11f_autodl_no_training_r02.sh"
    ):
        errors.append("R02 remote preflight script path mismatch")
    script_path = repo_root / str(preflight.get("script", ""))
    script_sha = preflight.get("script_sha256")
    if not isinstance(script_sha, str) or not SHA256_RE.fullmatch(script_sha):
        errors.append("preflight script SHA-256 invalid")
    if not script_path.is_file():
        errors.append("preflight script missing")
    else:
        if (
            isinstance(script_sha, str)
            and SHA256_RE.fullmatch(script_sha)
            and _sha256_lf_normalized(script_path) != script_sha
        ):
            errors.append("preflight script SHA-256 mismatch")
        source = script_path.read_text(encoding="utf-8")
        fixed_paths = {
            "B11F_BUNDLE": bundle.get("remote_path"),
            "B11F_REPO": preflight.get("checkout_path"),
            "B11F_FREEZE_DIR": str(
                inputs.get("b01_freeze_manifest", {}).get("remote_path", "")
            ).rsplit("/", 1)[0],
            "B11F_DATA_ROOT": inputs.get("remote_dataset_root"),
        }
        for name, expected_path in fixed_paths.items():
            assignment = f'readonly {name}="{expected_path}"'
            if not isinstance(expected_path, str) or not expected_path.startswith("/root/autodl-tmp/"):
                errors.append(f"{name} fixed remote path invalid")
            elif assignment not in source:
                errors.append(f"{name} fixed remote path mismatch")
        for token in FORBIDDEN_SCRIPT_TOKENS:
            if token in source:
                errors.append(f"preflight script contains forbidden token: {token}")
        if re.search(r"EXP-SLP-B11F", source):
            errors.append("preflight script must not contain a formal B11F EXP-ID")
        for required in ("--validate-only", "--environment-preflight", "TRAINING_NOT_STARTED", "TEST=0"):
            if required not in source:
                errors.append(f"preflight script missing required marker/command: {required}")

    test_gate = payload.get("test_gate", {})
    if test_gate != {"test_access": False, "test_rows": 0, "test_labels": 0, "test_onehot": 0}:
        errors.append("TEST gate must remain false/zero")
    if payload.get("next_gate") != "OWNER_AUTHORIZATION_FOR_EXACT_AUTODL_NO_TRAINING_PREFLIGHT_R02":
        errors.append("next gate mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    errors = validate(args.manifest.resolve(), ROOT)
    if errors:
        for error in errors:
            print(f"ERR: {error}")
        print("B11F_AUTODL_PREFLIGHT_PREPARATION_VALIDATION_FAILED")
        return 1
    print("summary: PASS (exact bundle/input hashes and no-training boundaries)")
    print("B11F_AUTODL_PREFLIGHT_PREPARATION_VALIDATION_PASSED TEST=0 AUTODL_R02_NOT_AUTHORIZED GPU_NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
