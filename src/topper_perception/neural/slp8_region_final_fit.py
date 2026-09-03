"""B11F final development-fit runner (TRAIN+VAL only, TEST denied)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import optim

from topper_perception.io.slp8_training_table_freeze import (
    EXPECTED_COVERS, EXPECTED_PROVENANCE, EXPECTED_REVIEW_STATUS,
    EXPECTED_SETTINGS, load_b01_freeze_tables, manifest_sha256,
)
from topper_perception.neural.slp8_region_determinism import apply_settings, environment_payload
from topper_perception.neural.slp8_region_dataset import RegionSample, Slp8RegionDataset, build_dataloader
from topper_perception.neural.slp8_region_full import (
    N_CLASSES,
    atomic_write_json,
    build_model,
    compute_fold_class_weights_from_samples,
    compute_fold_normalization_from_samples,
    deterministic_cross_entropy_2d,
)
from topper_perception.neural.slp8_region_class_weights import class_weights_to_tensor


TASK_ID = "TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-PREPARATION-v0.1"
PROTOCOL = "B11F_FINAL_DEVELOPMENT_FIT"
MODEL = "slp8_deeplabv3plus_lite_v0.1"
SEEDS = (42, 123, 2026)
EPOCHS = {42: 15, 123: 20, 2026: 12}
OPTIMIZER = "AdamW"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_TOTAL_WALL_SECONDS = 2700
EXP_ID_RE = re.compile(r"^EXP-SLP-B11F-PM-FINAL-FIT-\d{8}-AUTODL-R\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalFitError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalFitProtocol:
    path: Path
    sha256: str
    candidate_contract: Path
    model_family: str
    seeds: tuple[int, ...]
    epochs: dict[int, int]
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str
    max_peak_cuda_mb: int
    min_free_disk_bytes: int
    max_total_wall_seconds: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: Path, repo_root: Path) -> FinalFitProtocol:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FinalFitError(f"unreadable B11F protocol: {exc}") from exc
    errors: list[str] = []
    pool, training, outputs, resources = raw.get("development_pool", {}), raw.get("training", {}), raw.get("outputs", {}), raw.get("resources", {})
    if raw.get("protocol") != PROTOCOL: errors.append("protocol mismatch")
    if raw.get("status") != "PREPARATION_ONLY_GPU_NOT_AUTHORIZED": errors.append("status must remain preparation-only")
    if raw.get("execution_authorized") is not False: errors.append("execution_authorized must be strict false")
    if raw.get("model_family") != MODEL: errors.append("model family mismatch")
    if pool.get("splits") != ["train", "val"]: errors.append("development splits mismatch")
    if pool.get("subjects") != 91 or pool.get("samples") != 4095: errors.append("development cardinality mismatch")
    if pool.get("test_access") is not False or pool.get("test_rows") != 0: errors.append("TEST must remain inaccessible and zero")
    if training.get("seeds") != list(SEEDS): errors.append("seed set mismatch")
    if training.get("fixed_epochs_by_seed") != {str(k): v for k, v in EPOCHS.items()}: errors.append("fixed epochs mismatch")
    if training.get("early_stopping") is not False: errors.append("early stopping must be false")
    if training.get("optimizer") != OPTIMIZER: errors.append("optimizer must inherit frozen AdamW")
    if training.get("batch_size") != BATCH_SIZE: errors.append("batch size mismatch")
    if training.get("learning_rate") != LEARNING_RATE: errors.append("learning rate mismatch")
    if training.get("weight_decay") != WEIGHT_DECAY: errors.append("weight decay mismatch")
    if training.get("shuffle") is not True: errors.append("shuffle must be strict true")
    if outputs.get("checkpoint_name") != "final.pt": errors.append("checkpoint name mismatch")
    if resources.get("max_peak_cuda_mb") != 8192 or resources.get("min_free_disk_bytes") != 1073741824 or resources.get("max_total_wall_seconds") != MAX_TOTAL_WALL_SECONDS: errors.append("resource contract mismatch")
    if outputs.get("models_required") != 3 or outputs.get("training_metrics_are_not_validation") is not True: errors.append("output contract mismatch")
    if raw.get("test_gate") != "B09T_SEPARATE_ONE_TIME_OWNER_AUTHORIZATION_REQUIRED": errors.append("TEST gate mismatch")
    candidate_rel = raw.get("candidate_contract")
    candidate_path = repo_root / str(candidate_rel)
    if not candidate_path.is_file(): errors.append("candidate contract missing")
    else:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        fit = candidate.get("final_development_fit", {})
        if candidate.get("model_family") != MODEL: errors.append("candidate model mismatch")
        if fit.get("seeds") != list(SEEDS) or fit.get("fixed_epochs_by_seed") != {str(k): v for k, v in EPOCHS.items()}: errors.append("candidate final-fit freeze mismatch")
        if candidate.get("development_evidence", {}).get("test_access") is not False: errors.append("candidate TEST access mismatch")
    if errors:
        raise FinalFitError("; ".join(errors))
    return FinalFitProtocol(
        path=path, sha256=sha256_file(path), candidate_contract=candidate_path,
        model_family=MODEL, seeds=SEEDS, epochs=EPOCHS,
        batch_size=int(training["batch_size"]), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]), optimizer=OPTIMIZER,
        max_peak_cuda_mb=8192, min_free_disk_bytes=1073741824,
        max_total_wall_seconds=MAX_TOTAL_WALL_SECONDS,
    )


def build_plan(protocol: FinalFitProtocol) -> tuple[tuple[int, int], ...]:
    return tuple((seed, protocol.epochs[seed]) for seed in protocol.seeds)


def load_development_samples(freeze_dir: Path) -> list[RegionSample]:
    freeze = load_b01_freeze_tables(freeze_dir, load_test=False)
    if freeze._test_rows is not None:  # type: ignore[attr-defined]
        raise FinalFitError("B11F loader observed TEST rows; refusing")
    core = freeze.freeze_manifest.get("core", {})
    splits = core.get("splits", {})
    if freeze.train_manifest_sha256 != splits.get("train", {}).get("manifest_sha256"):
        raise FinalFitError("TRAIN manifest hash does not match freeze manifest")
    if freeze.val_manifest_sha256 != splits.get("val", {}).get("manifest_sha256"):
        raise FinalFitError("VAL manifest hash does not match freeze manifest")
    if len(freeze.train_rows) != 3645 or len(freeze.val_rows) != 450:
        raise FinalFitError("B11F requires exact TRAIN=3645 / VAL=450")
    train_subjects = {r.subject_id for r in freeze.train_rows}
    val_subjects = {r.subject_id for r in freeze.val_rows}
    if len(train_subjects) != 81 or len(val_subjects) != 10 or train_subjects & val_subjects:
        raise FinalFitError("development subject split mismatch or overlap")
    for row in [*freeze.train_rows, *freeze.val_rows]:
        if row.setting not in EXPECTED_SETTINGS or row.cover not in EXPECTED_COVERS:
            raise FinalFitError("unexpected setting or cover in B01 development rows")
        if row.annotation_provenance != EXPECTED_PROVENANCE or row.source_review_status != EXPECTED_REVIEW_STATUS:
            raise FinalFitError("B01 provenance/review contract mismatch")
        if not row.onehot_valid or not row.onehot_roundtrip:
            raise FinalFitError("B01 development onehot contract mismatch")
        path_parts = [part.lower() for value in (row.pressure_npy, row.region_label_npy, row.region_onehot_npy) for part in Path(value).parts]
        if any(part in {"test", "test_like"} for part in path_parts):
            raise FinalFitError("development row path contains forbidden TEST-like segment")
    rows = [*freeze.train_rows, *freeze.val_rows]
    samples = [RegionSample(r.sample_id, r.subject_id, r.ml_split, r.posture, r.pressure_npy, r.region_label_npy, r.region_onehot_npy) for r in rows]
    if len(samples) != 4095 or len({s.subject_id for s in samples}) != 91 or len({s.sample_id for s in samples}) != 4095:
        raise FinalFitError("development pool must be exactly 4095 samples / 91 subjects")
    if any(s.ml_split not in {"train", "val"} for s in samples):
        raise FinalFitError("non-development split entered final fit")
    return samples


def _identity(
    protocol: FinalFitProtocol,
    experiment_id: str,
    git_commit: str,
    git_dirty: bool,
    data_hash: str,
    candidate_hash: str,
    seed: int,
    epochs: int,
    *,
    authorized_environment_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "task_id": TASK_ID, "experiment_id": experiment_id, "git_commit": git_commit,
        "git_dirty": git_dirty, "config_sha256": protocol.sha256,
        "candidate_config_sha256": candidate_hash,
        "data_manifest_sha256": data_hash, "model_version": MODEL,
        "seed": seed, "fixed_epochs": epochs, "test_access": False,
        "test_rows": 0, "test_labels": 0, "test_onehot": 0,
        "max_total_wall_seconds": protocol.max_total_wall_seconds,
    }
    if authorized_environment_sha256 is not None:
        identity["authorized_environment_sha256"] = authorized_environment_sha256
    return identity


def _checkpoint_matches(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FinalFitError(f"checkpoint unreadable: {exc}") from exc
    actual = payload.get("identity", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise FinalFitError(f"checkpoint identity mismatch: {key}")
    return payload


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:
        raise FinalFitError(f"checkpoint RNG state could not be restored: {exc}") from exc


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current: raise FinalFitError("cannot find an existing parent for output path")
        current = current.parent
    return current


def _environment_record() -> dict[str, Any]:
    payload = environment_payload()
    payload.update({
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    })
    return payload


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FinalFitError(f"{description} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalFitError(f"{description} must be a JSON object")
    return payload


def _require_strict_json(payload: Mapping[str, Any], description: str) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FinalFitError(f"{description} is not strict JSON: {exc}") from exc


def _finite_nonnegative(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalFitError(f"{description} is missing or invalid") from exc
    if not np.isfinite(result) or result < 0:
        raise FinalFitError(f"{description} must be finite and non-negative")
    return result


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    _require_strict_json(payload, "canonical hash payload")
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_preflight_payload() -> dict[str, Any]:
    """Collect the exact no-training environment input for Owner authorization."""
    apply_settings(SEEDS[0])
    environment = _environment_record()
    payload = {
        "environment": environment,
        "environment_fingerprint_sha256": _canonical_json_sha256(environment),
        "test_access": False,
        "test_rows": 0,
        "gpu_training_run": False,
    }
    _require_strict_json(payload, "environment preflight payload")
    return payload


def _budget_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "identity": payload.get("identity"),
        "max_total_wall_seconds": payload.get("max_total_wall_seconds"),
        "started_at_utc_epoch_seconds": payload.get("started_at_utc_epoch_seconds"),
        "deadline_utc_epoch_seconds": payload.get("deadline_utc_epoch_seconds"),
    }


def _new_budget(identity: Mapping[str, Any], max_total_wall_seconds: int) -> tuple[dict[str, Any], str]:
    now = float(time.time())
    if not math.isfinite(now) or now < 0:
        raise FinalFitError("wall clock is invalid before budget creation")
    payload: dict[str, Any] = {
        "schema": "B11F_EXP_WALL_BUDGET_v0.1",
        "identity": dict(identity),
        "max_total_wall_seconds": int(max_total_wall_seconds),
        "started_at_utc_epoch_seconds": now,
        "deadline_utc_epoch_seconds": now + int(max_total_wall_seconds),
        "observed_at_utc_epoch_seconds": now,
        "elapsed_wall_seconds": 0.0,
        "remaining_wall_seconds": float(max_total_wall_seconds),
        "state": "RUNNING",
    }
    core_sha = _canonical_json_sha256(_budget_core(payload))
    payload["budget_core_sha256"] = core_sha
    _require_strict_json(payload, "new experiment budget")
    return payload, core_sha


def _refresh_budget(
    path: Path,
    expected_identity: Mapping[str, Any],
    expected_core_sha256: str,
    *,
    state: str,
    fail_if_exhausted: bool = True,
) -> dict[str, Any]:
    payload = _read_json_object(path, "experiment wall budget")
    if payload.get("schema") != "B11F_EXP_WALL_BUDGET_v0.1":
        raise FinalFitError("experiment wall budget schema mismatch")
    if payload.get("identity") != dict(expected_identity):
        raise FinalFitError("experiment wall budget identity mismatch")
    if payload.get("max_total_wall_seconds") != MAX_TOTAL_WALL_SECONDS:
        raise FinalFitError("experiment wall budget maximum mismatch")
    if not SHA256_RE.fullmatch(str(expected_core_sha256)):
        raise FinalFitError("experiment wall budget carrier hash is invalid")
    observed_core_sha = _canonical_json_sha256(_budget_core(payload))
    if payload.get("budget_core_sha256") != observed_core_sha or observed_core_sha != expected_core_sha256:
        raise FinalFitError("experiment wall budget immutable core mismatch")
    started = _finite_nonnegative(payload.get("started_at_utc_epoch_seconds"), "budget start time")
    deadline = _finite_nonnegative(payload.get("deadline_utc_epoch_seconds"), "budget deadline")
    observed = _finite_nonnegative(payload.get("observed_at_utc_epoch_seconds"), "budget observation time")
    elapsed = _finite_nonnegative(payload.get("elapsed_wall_seconds"), "budget elapsed wall time")
    remaining = _finite_nonnegative(payload.get("remaining_wall_seconds"), "budget remaining wall time")
    if abs(deadline - (started + MAX_TOTAL_WALL_SECONDS)) > 1e-6:
        raise FinalFitError("experiment wall budget deadline drift")
    if observed < started or abs(elapsed - (observed - started)) > 1e-3:
        raise FinalFitError("experiment wall budget elapsed carrier mismatch")
    if abs(remaining - max(0.0, deadline - observed)) > 1e-3:
        raise FinalFitError("experiment wall budget remaining carrier mismatch")
    now = float(time.time())
    if not math.isfinite(now) or now + 1e-6 < observed:
        raise FinalFitError("wall clock moved backwards during experiment")
    payload.update({
        "observed_at_utc_epoch_seconds": now,
        "elapsed_wall_seconds": now - started,
        "remaining_wall_seconds": max(0.0, deadline - now),
        "state": state,
    })
    _require_strict_json(payload, "experiment wall budget")
    atomic_write_json(path, payload)
    if fail_if_exhausted and payload["remaining_wall_seconds"] <= 0:
        raise FinalFitError("experiment total wall budget exhausted")
    return payload


def _budget_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "budget_core_sha256": payload.get("budget_core_sha256"),
        "max_total_wall_seconds": payload.get("max_total_wall_seconds"),
        "started_at_utc_epoch_seconds": payload.get("started_at_utc_epoch_seconds"),
        "deadline_utc_epoch_seconds": payload.get("deadline_utc_epoch_seconds"),
        "elapsed_wall_seconds": payload.get("elapsed_wall_seconds"),
        "remaining_wall_seconds": payload.get("remaining_wall_seconds"),
        "state": payload.get("state"),
    }


def _current_peak_cuda_mb(device: str) -> float:
    return float(torch.cuda.max_memory_allocated()) / 1e6 if device == "cuda" else 0.0


def _validate_resume_environment(carrier: Mapping[str, Any], environment_file: Path) -> str:
    if carrier.get("environment_path") != "environment.json":
        raise FinalFitError("resume environment evidence path mismatch")
    expected_sha = carrier.get("environment_sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise FinalFitError("resume environment evidence SHA is missing or invalid")
    if not environment_file.is_file() or sha256_file(environment_file) != expected_sha:
        raise FinalFitError("resume environment evidence hash mismatch")
    original_environment = _read_json_object(environment_file, "environment evidence")
    if _environment_record() != original_environment:
        raise FinalFitError("resume environment differs from original dispatch environment")
    return expected_sha


def _reconcile_interrupted_terminal_transition(output_dir: Path) -> None:
    """Finish an interrupted RUNNING->terminal rename without creating two states."""
    running = output_dir / "RUNNING.json"
    if not running.is_file():
        return
    payload = _read_json_object(running, "RUNNING carrier")
    state = payload.get("terminal_state")
    if state == "RUNNING":
        return
    if state not in {"STOPPED", "FAILED", "DONE"}:
        raise FinalFitError("RUNNING carrier has an invalid terminal state")
    target = output_dir / f"{state}.json"
    if target.exists():
        raise FinalFitError("contradictory root state carriers detected")
    os.replace(running, target)


def _write_terminal(output_dir: Path, name: str, payload: dict[str, Any]) -> None:
    """Persist one root state, then atomically rename RUNNING to its terminal name."""
    if name not in {"STOPPED.json", "FAILED.json", "DONE.json"}:
        raise FinalFitError("invalid terminal state filename")
    running = output_dir / "RUNNING.json"
    if not running.is_file():
        raise FinalFitError("RUNNING carrier missing during terminal transition")
    if any((output_dir / item).exists() for item in ("STOPPED.json", "FAILED.json", "DONE.json")):
        raise FinalFitError("terminal carrier already exists")
    _require_strict_json(payload, f"{name} payload")
    atomic_write_json(running, payload)
    os.replace(running, output_dir / name)


def run_final_fit(*, protocol: FinalFitProtocol, freeze_dir: Path, data_root: Path, output_dir: Path, experiment_id: str, git_commit: str, git_dirty: bool, authorized_environment_sha256: str | None = None, device: str = "cuda", resume: bool = False) -> dict[str, Any]:
    if not EXP_ID_RE.fullmatch(experiment_id): raise FinalFitError("invalid B11F EXP-ID")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit) or git_dirty is not False: raise FinalFitError("real final fit requires a clean frozen 40-char Git SHA")
    if not isinstance(authorized_environment_sha256, str) or not SHA256_RE.fullmatch(authorized_environment_sha256):
        raise FinalFitError("real final fit requires a 64-char authorized environment fingerprint")
    # Establish cuBLAS environment before *any* CUDA probe.
    apply_settings(SEEDS[0])
    observed_environment = _environment_record()
    observed_environment_fingerprint = _canonical_json_sha256(observed_environment)
    if observed_environment_fingerprint != authorized_environment_sha256:
        raise FinalFitError("current environment does not match Owner-authorized fingerprint")
    if output_dir.exists() and resume:
        _reconcile_interrupted_terminal_transition(output_dir)
    if output_dir.exists() and any((output_dir / name).exists() for name in ("DONE.json", "FAILED.json")):
        raise FinalFitError("terminal output is immutable")
    if output_dir.exists() and (output_dir / "RUNNING.json").is_file() and (output_dir / "STOPPED.json").is_file():
        raise FinalFitError("contradictory resumable root states detected")
    if output_dir.exists() and (output_dir / "RUNNING.json").is_file() and not resume: raise FinalFitError("existing RUNNING output requires explicit resume authorization")
    if output_dir.exists() and not (resume and ((output_dir / "RUNNING.json").is_file() or (output_dir / "STOPPED.json").is_file())): raise FinalFitError("output directory already exists; completed evidence is immutable")
    if device == "cuda" and not torch.cuda.is_available(): raise FinalFitError("CUDA is required but unavailable")
    if shutil.disk_usage(_existing_parent(output_dir.parent)).free < protocol.min_free_disk_bytes:
        raise FinalFitError("insufficient free disk space for final fit")
    manifest_path = freeze_dir / "freeze_manifest.json"
    if not manifest_path.is_file(): raise FinalFitError("B01 freeze manifest missing")
    data_hash = sha256_file(manifest_path)
    candidate_hash = sha256_file(protocol.candidate_contract)
    run_identity = _identity(protocol, experiment_id, git_commit, git_dirty, data_hash, candidate_hash, 0, 0, authorized_environment_sha256=authorized_environment_sha256)
    run_identity.pop("seed"); run_identity.pop("fixed_epochs")
    running_path = output_dir / "RUNNING.json"
    environment_file = output_dir / "environment.json"
    budget_file = output_dir / "budget.json"
    if resume and (output_dir / "STOPPED.json").is_file():
        stopped_path = output_dir / "STOPPED.json"
        stopped = _read_json_object(stopped_path, "STOPPED carrier")
        if stopped.get("identity") != run_identity: raise FinalFitError("resume STOPPED identity mismatch")
        environment_sha = _validate_resume_environment(stopped, environment_file)
        budget_core_sha = str(stopped.get("budget_core_sha256", ""))
        budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="STOPPED")
        budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
        running_payload = {"terminal_state": "RUNNING", "identity": run_identity, "environment_path": "environment.json", "environment_sha256": environment_sha, "budget_path": "budget.json", "budget_core_sha256": budget_core_sha, "budget": _budget_summary(budget), "resumed_from": "STOPPED.json"}
        atomic_write_json(stopped_path, running_payload)
        os.replace(stopped_path, running_path)
    elif running_path.is_file():
        if not resume: raise FinalFitError("existing RUNNING output requires explicit resume authorization")
        persisted = _read_json_object(running_path, "RUNNING carrier")
        if persisted.get("terminal_state") != "RUNNING": raise FinalFitError("RUNNING carrier state mismatch")
        if persisted.get("identity") != run_identity: raise FinalFitError("resume RUNNING identity mismatch")
        environment_sha = _validate_resume_environment(persisted, environment_file)
        budget_core_sha = str(persisted.get("budget_core_sha256", ""))
        budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
        atomic_write_json(running_path, {"terminal_state": "RUNNING", "identity": run_identity, "environment_path": "environment.json", "environment_sha256": environment_sha, "budget_path": "budget.json", "budget_core_sha256": budget_core_sha, "budget": _budget_summary(budget), "resumed_from": "RUNNING.json"})
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(environment_file, observed_environment)
        environment_sha = sha256_file(environment_file)
        budget, budget_core_sha = _new_budget(run_identity, protocol.max_total_wall_seconds)
        atomic_write_json(budget_file, budget)
        atomic_write_json(running_path, {"terminal_state": "RUNNING", "identity": run_identity, "environment_path": "environment.json", "environment_sha256": environment_sha, "budget_path": "budget.json", "budget_core_sha256": budget_core_sha, "budget": _budget_summary(budget)})
    results: list[dict[str, Any]] = []
    try:
        budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
        samples = load_development_samples(freeze_dir)
        normalization = compute_fold_normalization_from_samples(samples, data_root=data_root)
        class_weights = compute_fold_class_weights_from_samples(samples, data_root=data_root)
        dataset = Slp8RegionDataset(samples, data_root, normalization)
        for seed, epochs in build_plan(protocol):
            settings = apply_settings(seed)
            if device == "cuda": torch.cuda.reset_peak_memory_stats()
            seed_dir = output_dir / f"seed_{seed:04d}"; seed_dir.mkdir(exist_ok=True)
            ident = _identity(protocol, experiment_id, git_commit, git_dirty, data_hash, candidate_hash, seed, epochs, authorized_environment_sha256=authorized_environment_sha256)
            ident["environment_sha256"] = environment_sha
            ident["budget_core_sha256"] = budget_core_sha
            final_path, last_path = seed_dir / "final.pt", seed_dir / "last.pt"
            complete_path = seed_dir / "complete.json"
            if complete_path.is_file():
                prior = _read_json_object(complete_path, f"seed {seed} completion carrier")
                if prior.get("identity") != ident: raise FinalFitError(f"seed {seed} completed identity mismatch")
                if Path(str(prior.get("checkpoint"))).resolve() != final_path.resolve(): raise FinalFitError(f"seed {seed} checkpoint path mismatch")
                if not final_path.is_file() or prior.get("checkpoint_sha256") != sha256_file(final_path): raise FinalFitError(f"seed {seed} checkpoint SHA mismatch")
                _checkpoint_matches(final_path, ident); results.append(prior); continue
            if final_path.exists():
                raise FinalFitError(f"seed {seed} final checkpoint exists without completion carrier; refusing overwrite")
            model = build_model(MODEL, device)
            optimizer = optim.AdamW(model.parameters(), lr=protocol.lr, weight_decay=protocol.weight_decay)
            weights = torch.from_numpy(
                class_weights_to_tensor(class_weights)
            ).to(device).to(torch.float32)
            loader = build_dataloader(dataset, batch_size=protocol.batch_size, shuffle=True)
            segment_started = time.monotonic()
            accumulated_wall_seconds = 0.0
            historical_peak_cuda_mb = 0.0
            last_loss: float | None = None
            audit_inputs = next(iter(build_dataloader(dataset, batch_size=min(2, protocol.batch_size), shuffle=False)))["pressure"].to(device)
            start_epoch = 1
            if last_path.is_file():
                resumed = _checkpoint_matches(last_path, ident)
                model.load_state_dict(resumed["model_state_dict"]); optimizer.load_state_dict(resumed["optimizer_state_dict"])
                start_epoch = int(resumed["epoch"]) + 1
                if start_epoch < 2 or start_epoch > epochs + 1: raise FinalFitError(f"invalid resumable epoch for seed {seed}")
                last_loss = _finite_nonnegative(resumed.get("training_loss_last_epoch"), f"seed {seed} resumable training loss")
                accumulated_wall_seconds = _finite_nonnegative(resumed.get("elapsed_wall_seconds"), f"seed {seed} resumable wall time")
                historical_peak_cuda_mb = _finite_nonnegative(resumed.get("peak_cuda_mb"), f"seed {seed} resumable CUDA peak")
                if historical_peak_cuda_mb > protocol.max_peak_cuda_mb:
                    raise FinalFitError(f"CUDA peak memory exceeded for seed {seed}")
                _restore_rng_state(resumed.get("rng_state", {}))
            for _epoch in range(start_epoch, epochs + 1):
                model.train()
                losses = []
                for batch in loader:
                    budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
                    x, y = batch["pressure"].to(device), batch["label"].to(device)
                    optimizer.zero_grad(); loss = deterministic_cross_entropy_2d(model(x), y, weight=weights)
                    if not torch.isfinite(loss): raise FinalFitError(f"non-finite loss for seed {seed}")
                    loss.backward(); optimizer.step(); losses.append(float(loss.item()))
                    budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
                    if _current_peak_cuda_mb(device) > protocol.max_peak_cuda_mb:
                        raise FinalFitError(f"CUDA peak memory exceeded for seed {seed}")
                last_loss = float(np.mean(losses))
                if not np.isfinite(last_loss) or last_loss < 0:
                    raise FinalFitError(f"non-finite last epoch loss for seed {seed}")
                elapsed_wall_seconds = accumulated_wall_seconds + (time.monotonic() - segment_started)
                peak_cuda_mb = max(historical_peak_cuda_mb, _current_peak_cuda_mb(device))
                tmp_last = seed_dir / "last.pt.tmp"
                budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": _epoch, "training_loss_last_epoch": last_loss, "elapsed_wall_seconds": elapsed_wall_seconds, "peak_cuda_mb": peak_cuda_mb, "rng_state": _capture_rng_state(), "experiment_budget": _budget_summary(budget), "identity": ident}, tmp_last)
                os.replace(tmp_last, last_path)
            if last_loss is None or not np.isfinite(last_loss) or last_loss < 0:
                raise FinalFitError(f"seed {seed} has no finite final training loss")
            model.eval()
            with torch.no_grad(): before = model(audit_inputs).argmax(1).cpu()
            tmp_path = seed_dir / "final.pt.tmp"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epochs, "identity": ident}, tmp_path)
            os.replace(tmp_path, final_path)
            loaded = _checkpoint_matches(final_path, ident)
            reloaded = build_model(MODEL, device); reloaded.load_state_dict(loaded["model_state_dict"]); reloaded.eval()
            with torch.no_grad(): after = reloaded(audit_inputs).argmax(1).cpu()
            if not torch.equal(before, after): raise FinalFitError(f"reload prediction mismatch for seed {seed}")
            final_peak_cuda_mb = max(historical_peak_cuda_mb, _current_peak_cuda_mb(device))
            if final_peak_cuda_mb > protocol.max_peak_cuda_mb:
                raise FinalFitError(f"CUDA peak memory exceeded for seed {seed}")
            budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="RUNNING")
            result = {"seed": seed, "fixed_epochs": epochs, "training_loss_last_epoch": last_loss, "training_loss_is_not_validation": True, "wall_seconds": accumulated_wall_seconds + (time.monotonic() - segment_started), "peak_cuda_mb": final_peak_cuda_mb, "checkpoint": str(final_path), "checkpoint_sha256": sha256_file(final_path), "reload_prediction_match": True, "determinism": settings.as_dict(), "experiment_budget": _budget_summary(budget), "identity": ident}
            _require_strict_json(result, f"seed {seed} completion payload")
            atomic_write_json(seed_dir / "complete.json", result); results.append(result)
        if len(results) != 3:
            raise FinalFitError(f"three final models required, got {len(results)}")
        for result in results:
            seed = int(result["seed"])
            final_path = output_dir / f"seed_{seed:04d}" / "final.pt"
            if Path(str(result.get("checkpoint"))).resolve() != final_path.resolve():
                raise FinalFitError("final checkpoint path audit failed before DONE")
            if not final_path.is_file() or sha256_file(final_path) != result["checkpoint_sha256"]:
                raise FinalFitError("final checkpoint audit failed before DONE")
            _checkpoint_matches(final_path, result["identity"])
        if not environment_file.is_file(): raise FinalFitError("environment evidence missing before DONE")
        running = _read_json_object(running_path, "RUNNING carrier")
        environment_sha = sha256_file(environment_file)
        if running.get("environment_sha256") != environment_sha:
            raise FinalFitError("environment evidence changed during run")
        budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state="DONE")
        budget_sha = sha256_file(budget_file)
        done = {"terminal_state": "DONE", "identity": run_identity, "environment_path": "environment.json", "environment_sha256": environment_sha, "budget_path": "budget.json", "budget_sha256": budget_sha, "budget_core_sha256": budget_core_sha, "budget": _budget_summary(budget), "models_complete": len(results), "models_required": 3, "results": results}
        _require_strict_json(done, "DONE payload")
        _write_terminal(output_dir, "DONE.json", done)
        return done
    except BaseException as exc:
        state = "STOPPED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        observed_environment_sha = sha256_file(environment_file) if environment_file.is_file() else None
        budget_error = None
        try:
            budget = _refresh_budget(budget_file, run_identity, budget_core_sha, state=state, fail_if_exhausted=False)
        except BaseException as budget_exc:
            budget_error = f"{type(budget_exc).__name__}: {budget_exc}"
        observed_budget_sha = sha256_file(budget_file) if budget_file.is_file() else None
        failed = {
            "terminal_state": state,
            "identity": run_identity,
            "environment_path": "environment.json",
            "environment_sha256": environment_sha,
            "observed_environment_sha256": observed_environment_sha,
            "environment_hash_match": observed_environment_sha == environment_sha,
            "budget_path": "budget.json",
            "budget_sha256": observed_budget_sha,
            "budget_core_sha256": budget_core_sha,
            "budget": _budget_summary(budget) if budget_error is None else None,
            "budget_error": budget_error,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_terminal(output_dir, f"{state}.json", failed)
        raise
