"""B11F final development-fit preparation tests (no GPU, no TEST)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import topper_perception.neural.slp8_region_final_fit as ff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/slp8_pm_final_development_fit_v0.1.json"


def test_frozen_protocol_and_plan():
    p = ff.load_protocol(CONFIG, ROOT)
    assert ff.build_plan(p) == ((42, 15), (123, 20), (2026, 12))
    assert p.model_family == ff.MODEL


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["development_pool"].__setitem__("test_access", True), "TEST"),
    (lambda d: d["development_pool"].__setitem__("splits", ["train", "val", "test"]), "splits"),
    (lambda d: d["training"]["fixed_epochs_by_seed"].__setitem__("42", 16), "epochs"),
    (lambda d: d.__setitem__("execution_authorized", True), "execution_authorized"),
    (lambda d: d["training"].__setitem__("optimizer", "SGD"), "optimizer"),
    (lambda d: d["training"].__setitem__("batch_size", 1), "batch size"),
    (lambda d: d["training"].__setitem__("learning_rate", 9), "learning rate"),
    (lambda d: d["training"].__setitem__("weight_decay", 7), "weight decay"),
])
def test_protocol_drift_fails_closed(tmp_path: Path, mutation, match: str):
    d = json.loads(CONFIG.read_text(encoding="utf-8")); mutation(d)
    path = tmp_path / "bad.json"; path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ff.FinalFitError, match=match): ff.load_protocol(path, ROOT)


def test_loader_combines_only_development_and_test_is_unloaded(monkeypatch):
    def row(i, split):
        subject = i % 81 if split == "train" else 81 + (i % 10)
        return SimpleNamespace(sample_id=f"s{i}", subject_id=f"{subject:05d}", ml_split=split, posture="supine", pressure_npy="p.npy", region_label_npy="l.npy", region_onehot_npy="o.npy", setting="danaLab", cover="uncover", annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED", source_review_status="NOT_REVIEWED", onehot_valid=True, onehot_roundtrip=True)
    fake = SimpleNamespace(train_rows=[row(i, "train") for i in range(3645)], val_rows=[row(i, "val") for i in range(3645, 4095)], _test_rows=None)
    calls = []
    fake.train_manifest_sha256 = "train"; fake.val_manifest_sha256 = "val"; fake.freeze_manifest = {"core": {"splits": {"train": {"manifest_sha256": "train"}, "val": {"manifest_sha256": "val"}}}}
    monkeypatch.setattr(ff, "load_b01_freeze_tables", lambda path, load_test: calls.append(load_test) or fake)
    samples = ff.load_development_samples(Path("unused"))
    assert len(samples) == 4095 and len({s.subject_id for s in samples}) == 91
    assert calls == [False]


def test_loader_rejects_test_carrier(monkeypatch):
    fake = SimpleNamespace(train_rows=[], val_rows=[], _test_rows=[object()])
    monkeypatch.setattr(ff, "load_b01_freeze_tables", lambda path, load_test: fake)
    with pytest.raises(ff.FinalFitError, match="TEST"): ff.load_development_samples(Path("unused"))


def test_loader_rejects_test_like_path(monkeypatch):
    def row(i, split):
        subject = i % 81 if split == "train" else 81 + (i % 10)
        return SimpleNamespace(sample_id=f"s{i}", subject_id=f"{subject:05d}", ml_split=split, posture="supine", pressure_npy="test_like/p.npy" if i == 0 else "p.npy", region_label_npy="l.npy", region_onehot_npy="o.npy", setting="danaLab", cover="uncover", annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED", source_review_status="NOT_REVIEWED", onehot_valid=True, onehot_roundtrip=True)
    fake = SimpleNamespace(train_rows=[row(i, "train") for i in range(3645)], val_rows=[row(i, "val") for i in range(3645, 4095)], _test_rows=None, train_manifest_sha256="train", val_manifest_sha256="val", freeze_manifest={"core": {"splits": {"train": {"manifest_sha256": "train"}, "val": {"manifest_sha256": "val"}}}})
    monkeypatch.setattr(ff, "load_b01_freeze_tables", lambda path, load_test: fake)
    with pytest.raises(ff.FinalFitError, match="TEST-like"):
        ff.load_development_samples(Path("unused"))


def test_checkpoint_identity_exact(tmp_path: Path):
    path = tmp_path / "x.pt"; torch.save({"identity": {"seed": 42}, "model_state_dict": {}}, path)
    assert ff._checkpoint_matches(path, {"seed": 42})["identity"]["seed"] == 42
    with pytest.raises(ff.FinalFitError, match="identity mismatch"):
        ff._checkpoint_matches(path, {"seed": 123})


def test_real_run_rejects_bad_exp_before_output(tmp_path: Path):
    protocol = ff.load_protocol(CONFIG, ROOT); out = tmp_path / "out"
    with pytest.raises(ff.FinalFitError, match="EXP-ID"):
        ff.run_final_fit(protocol=protocol, freeze_dir=tmp_path, data_root=tmp_path, output_dir=out, experiment_id="BAD", git_commit="a" * 40, git_dirty=False)
    assert not out.exists()


def test_real_run_rejects_dirty_git_before_output(tmp_path: Path):
    protocol = ff.load_protocol(CONFIG, ROOT); out = tmp_path / "out"
    with pytest.raises(ff.FinalFitError, match="clean frozen"):
        ff.run_final_fit(protocol=protocol, freeze_dir=tmp_path, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=True)
    assert not out.exists()


def test_real_run_refuses_existing_output(tmp_path: Path):
    protocol = ff.load_protocol(CONFIG, ROOT); out = tmp_path / "out"; out.mkdir()
    with pytest.raises(ff.FinalFitError, match="already exists"):
        ff.run_final_fit(protocol=protocol, freeze_dir=tmp_path, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False)


def test_three_seed_success_writes_identity_and_reload_evidence(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    sample = SimpleNamespace()
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [sample])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "class_weights_to_tensor", lambda value: torch.ones(9))
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    batches = [{"pressure": torch.ones(1, 1, 4, 4), "label": torch.zeros(1, 4, 4, dtype=torch.long)}]
    monkeypatch.setattr(ff, "build_dataloader", lambda *a, **k: batches)
    monkeypatch.setattr(ff, "build_plan", lambda p: ((42, 1), (123, 1), (2026, 1)))
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: torch.nn.Conv2d(1, 9, 1))
    out = tmp_path / "out"
    done = ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    assert done["models_complete"] == 3
    assert done["identity"]["test_access"] is False
    assert done["identity"]["test_rows"] == 0
    assert not (out / "RUNNING.json").exists()
    assert (out / "DONE.json").is_file()
    done_disk = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
    assert done_disk["environment_sha256"] == ff.sha256_file(out / "environment.json")
    for seed in (42, 123, 2026):
        result = json.loads((out / f"seed_{seed:04d}" / "complete.json").read_text(encoding="utf-8"))
        assert result["reload_prediction_match"] is True
        assert result["training_loss_is_not_validation"] is True
        assert result["identity"]["candidate_config_sha256"] == done["identity"]["candidate_config_sha256"]


def test_failure_writes_failed_terminal(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: (_ for _ in ()).throw(ff.FinalFitError("injected")))
    out = tmp_path / "out"
    with pytest.raises(ff.FinalFitError, match="injected"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    failed = json.loads((out / "FAILED.json").read_text(encoding="utf-8"))
    assert failed["terminal_state"] == "FAILED"
    assert failed["identity"]["test_access"] is False


def test_keyboard_interrupt_writes_stopped_terminal(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    assert (out / "STOPPED.json").is_file()
    assert not (out / "RUNNING.json").exists()


def test_existing_running_requires_explicit_resume(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    out = tmp_path / "out"; out.mkdir()
    identity = ff._identity(protocol, "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", "a" * 40, False, ff.sha256_file(freeze / "freeze_manifest.json"), ff.sha256_file(protocol.candidate_contract), 0, 0); identity.pop("seed"); identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "RUNNING.json", {"identity": identity})
    with pytest.raises(ff.FinalFitError, match="explicit resume"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")


def test_stopped_resume_recreates_running_before_execution(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = tmp_path / "out"; out.mkdir()
    identity = ff._identity(protocol, "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", "a" * 40, False, ff.sha256_file(freeze / "freeze_manifest.json"), ff.sha256_file(protocol.candidate_contract), 0, 0); identity.pop("seed"); identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "environment.json", ff._environment_record())
    ff.atomic_write_json(out / "STOPPED.json", {"terminal_state": "STOPPED", "identity": identity, "environment_path": "environment.json", "environment_sha256": ff.sha256_file(out / "environment.json")})
    with pytest.raises(KeyboardInterrupt):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu", resume=True)
    assert (out / "STOPPED.json").is_file()
    assert not (out / "FAILED.json").exists()


def test_stopped_resume_rejects_tampered_environment(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    out = tmp_path / "out"; out.mkdir()
    identity = ff._identity(protocol, "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", "a" * 40, False, ff.sha256_file(freeze / "freeze_manifest.json"), ff.sha256_file(protocol.candidate_contract), 0, 0); identity.pop("seed"); identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "environment.json", {"original": True})
    original_sha = ff.sha256_file(out / "environment.json")
    ff.atomic_write_json(out / "STOPPED.json", {"terminal_state": "STOPPED", "identity": identity, "environment_path": "environment.json", "environment_sha256": original_sha})
    ff.atomic_write_json(out / "environment.json", {"tampered": True})
    with pytest.raises(ff.FinalFitError, match="environment evidence hash mismatch"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu", resume=True)
    assert (out / "STOPPED.json").is_file()


def test_resumed_training_matches_uninterrupted_parameters(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    sample = object(); batches = [{"pressure": torch.ones(1, 1, 4, 4), "label": torch.zeros(1, 4, 4, dtype=torch.long)}]
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [sample])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "class_weights_to_tensor", lambda value: torch.ones(9))
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    monkeypatch.setattr(ff, "build_dataloader", lambda *a, **k: batches)
    monkeypatch.setattr(ff, "build_plan", lambda p: ((42, 2), (123, 2), (2026, 2)))
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: torch.nn.Conv2d(1, 9, 1))
    original_loss = ff.deterministic_cross_entropy_2d; calls = {"n": 0}
    def interrupt_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2: raise KeyboardInterrupt()
        return original_loss(*args, **kwargs)
    monkeypatch.setattr(ff, "deterministic_cross_entropy_2d", interrupt_once)
    interrupted = tmp_path / "interrupted"
    kwargs = dict(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    with pytest.raises(KeyboardInterrupt): ff.run_final_fit(output_dir=interrupted, **kwargs)
    monkeypatch.setattr(ff, "deterministic_cross_entropy_2d", original_loss)
    ff.run_final_fit(output_dir=interrupted, resume=True, **kwargs)
    baseline = tmp_path / "baseline"
    baseline_kwargs = {**kwargs, "experiment_id": "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R02"}
    ff.run_final_fit(output_dir=baseline, **baseline_kwargs)
    for seed in (42, 123, 2026):
        resumed = torch.load(interrupted / f"seed_{seed:04d}" / "final.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        full = torch.load(baseline / f"seed_{seed:04d}" / "final.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        assert all(torch.equal(resumed[k], full[k]) for k in resumed)


def test_running_resume_rejects_tampered_environment_before_training(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"; out.mkdir(); ff.apply_settings(42)
    identity = ff._identity(protocol, "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", "a" * 40, False, ff.sha256_file(freeze / "freeze_manifest.json"), ff.sha256_file(protocol.candidate_contract), 0, 0); identity.pop("seed"); identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "environment.json", ff._environment_record())
    original_sha = ff.sha256_file(out / "environment.json")
    ff.atomic_write_json(out / "RUNNING.json", {"terminal_state": "RUNNING", "identity": identity, "environment_path": "environment.json", "environment_sha256": original_sha})
    ff.atomic_write_json(out / "environment.json", {"tampered": True})
    monkeypatch.setattr(ff, "load_development_samples", lambda path: pytest.fail("training data loaded before environment rejection"))
    with pytest.raises(ff.FinalFitError, match="environment evidence hash mismatch"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu", resume=True)
    assert (out / "RUNNING.json").is_file()
    assert not (out / "FAILED.json").exists()


def test_existing_final_without_completion_is_never_overwritten(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    out = tmp_path / "out"; out.mkdir(); ff.apply_settings(42)
    identity = ff._identity(protocol, "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", "a" * 40, False, ff.sha256_file(freeze / "freeze_manifest.json"), ff.sha256_file(protocol.candidate_contract), 0, 0); identity.pop("seed"); identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "environment.json", ff._environment_record())
    environment_sha = ff.sha256_file(out / "environment.json")
    ff.atomic_write_json(out / "STOPPED.json", {"terminal_state": "STOPPED", "identity": identity, "environment_path": "environment.json", "environment_sha256": environment_sha})
    seed_dir = out / "seed_0042"; seed_dir.mkdir(); final_path = seed_dir / "final.pt"; final_path.write_bytes(b"crash-window-checkpoint")
    before = ff.sha256_file(final_path)
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: pytest.fail("model built before final checkpoint collision rejection"))
    with pytest.raises(ff.FinalFitError, match="refusing overwrite"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu", resume=True)
    assert ff.sha256_file(final_path) == before
    assert (out / "FAILED.json").is_file()
    assert not (out / "RUNNING.json").exists()


def test_terminal_rename_failure_never_creates_contradictory_states(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"; out.mkdir(); running = out / "RUNNING.json"
    ff.atomic_write_json(running, {"terminal_state": "RUNNING"})
    real_replace = ff.os.replace
    def fail_terminal_rename(source, destination):
        if Path(source) == running and Path(destination).name == "DONE.json":
            raise PermissionError("injected terminal rename failure")
        return real_replace(source, destination)
    monkeypatch.setattr(ff.os, "replace", fail_terminal_rename)
    with pytest.raises(PermissionError, match="terminal rename"):
        ff._write_terminal(out, "DONE.json", {"terminal_state": "DONE"})
    assert sorted(path.name for path in out.glob("*.json")) == ["RUNNING.json"]
    assert json.loads(running.read_text(encoding="utf-8"))["terminal_state"] == "DONE"
    monkeypatch.setattr(ff.os, "replace", real_replace)
    ff._reconcile_interrupted_terminal_transition(out)
    assert sorted(path.name for path in out.glob("*.json")) == ["DONE.json"]


def test_rng_restore_replays_real_dataloader_shuffle():
    ff.apply_settings(42)
    state = ff._capture_rng_state()
    dataset = torch.utils.data.TensorDataset(torch.arange(12))
    first = torch.cat([batch[0] for batch in torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)]).tolist()
    ff._restore_rng_state(state)
    second = torch.cat([batch[0] for batch in torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)]).tolist()
    assert first == second


def test_apply_settings_precedes_every_cuda_probe(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    events = []
    monkeypatch.setattr(ff, "apply_settings", lambda seed: events.append(("apply_settings", seed)) or SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(ff.torch.cuda, "is_available", lambda: events.append(("cuda_is_available", None)) or False)
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=tmp_path / "out", experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    assert events[0] == ("apply_settings", 42)
    assert next(i for i, event in enumerate(events) if event[0] == "cuda_is_available") > 0


def test_resume_after_final_epoch_restores_finite_cumulative_diagnostics(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "class_weights_to_tensor", lambda value: torch.ones(9))
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    batches = [{"pressure": torch.ones(1, 1, 1, 1), "label": torch.zeros(1, 1, 1, dtype=torch.long)}]
    monkeypatch.setattr(ff, "build_dataloader", lambda *a, **k: batches)
    monkeypatch.setattr(ff, "build_plan", lambda p: ((42, 1), (123, 1), (2026, 1)))
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: torch.nn.Conv2d(1, 9, 1))
    out = tmp_path / "out"; out.mkdir(); ff.apply_settings(42)
    data_sha = ff.sha256_file(freeze / "freeze_manifest.json")
    candidate_sha = ff.sha256_file(protocol.candidate_contract)
    environment_path = out / "environment.json"; ff.atomic_write_json(environment_path, ff._environment_record())
    environment_sha = ff.sha256_file(environment_path)
    experiment_id = "EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01"
    run_identity = ff._identity(protocol, experiment_id, "a" * 40, False, data_sha, candidate_sha, 0, 0); run_identity.pop("seed"); run_identity.pop("fixed_epochs")
    ff.atomic_write_json(out / "STOPPED.json", {"terminal_state": "STOPPED", "identity": run_identity, "environment_path": "environment.json", "environment_sha256": environment_sha})
    ff.apply_settings(42)
    model = torch.nn.Conv2d(1, 9, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.lr, weight_decay=protocol.weight_decay)
    identity = ff._identity(protocol, experiment_id, "a" * 40, False, data_sha, candidate_sha, 42, 1); identity["environment_sha256"] = environment_sha
    seed_dir = out / "seed_0042"; seed_dir.mkdir()
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": 1, "training_loss_last_epoch": 0.375, "elapsed_wall_seconds": 12.5, "peak_cuda_mb": 0.0, "rng_state": ff._capture_rng_state(), "identity": identity}, seed_dir / "last.pt")
    done = ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id=experiment_id, git_commit="a" * 40, git_dirty=False, device="cpu", resume=True)
    resumed = done["results"][0]
    assert resumed["training_loss_last_epoch"] == 0.375
    assert math.isfinite(resumed["wall_seconds"]) and resumed["wall_seconds"] >= 12.5
    assert resumed["peak_cuda_mb"] == 0.0
    json.loads((out / "DONE.json").read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"))


def test_strict_json_gate_rejects_nonfinite_values():
    with pytest.raises(ff.FinalFitError, match="not strict JSON"):
        ff._require_strict_json({"training_loss_last_epoch": float("nan")}, "completion payload")


def test_reload_audit_peak_is_checked_before_completion(tmp_path: Path, monkeypatch):
    protocol = ff.load_protocol(CONFIG, ROOT)
    freeze = tmp_path / "freeze"; freeze.mkdir(); (freeze / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ff, "load_development_samples", lambda path: [object()])
    monkeypatch.setattr(ff, "compute_fold_normalization_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "compute_fold_class_weights_from_samples", lambda *a, **k: object())
    monkeypatch.setattr(ff, "class_weights_to_tensor", lambda value: torch.ones(9))
    monkeypatch.setattr(ff, "Slp8RegionDataset", lambda *a, **k: object())
    batches = [{"pressure": torch.ones(1, 1, 1, 1), "label": torch.zeros(1, 1, 1, dtype=torch.long)}]
    monkeypatch.setattr(ff, "build_dataloader", lambda *a, **k: batches)
    monkeypatch.setattr(ff, "build_plan", lambda p: ((42, 1),))
    monkeypatch.setattr(ff, "build_model", lambda *a, **k: torch.nn.Conv2d(1, 9, 1))
    peaks = iter((0.0, 0.0, float(protocol.max_peak_cuda_mb + 1)))
    monkeypatch.setattr(ff, "_current_peak_cuda_mb", lambda device: next(peaks))
    out = tmp_path / "out"
    with pytest.raises(ff.FinalFitError, match="CUDA peak memory exceeded"):
        ff.run_final_fit(protocol=protocol, freeze_dir=freeze, data_root=tmp_path, output_dir=out, experiment_id="EXP-SLP-B11F-PM-FINAL-FIT-20260902-AUTODL-R01", git_commit="a" * 40, git_dirty=False, device="cpu")
    assert (out / "FAILED.json").is_file()
    assert not (out / "DONE.json").exists()
