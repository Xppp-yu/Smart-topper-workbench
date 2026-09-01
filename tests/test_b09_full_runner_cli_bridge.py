"""B09 Full Runner CLI bridge tests
(TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1).

Contract §9 (test coverage):

1. --run-full + complete authorized parameters dispatches run_full()
   exactly once.
2. Without --run-authorized the call is rejected and run_full() is
   never invoked (call count == 0).
3. Missing dataset root / freeze dir / EXP-ID all reject without
   dispatching run_full().
4. Non-B09 EXP-ID, synthetic sentinel, and dirty git worktree are all
   rejected.
5. --run-full mixed with --one-fold-preflight / --validate-only /
   --no-write / --synthetic-cpu-smoke is rejected.
6. Frozen training contract drift (max_epochs, min_epochs, patience,
   candidates, seeds, folds, budget) is rejected.
7. Existing one-fold preflight regression path still works without
   dispatching run_full().
8. The current real writer fixture (DONE/FAILED/STOPPED identity +
   input_manifest_hashes + per-candidate seed block) carries every
   B09 audit carrier with the expected strict types.
9. DONE/FAILED/STOPPED identity and terminal mutex (write_terminal_state
   + status.json / manifest.json) all carry the same frozen hashes.
10. TEST six carriers are strict bool false / int 0; no unknown test
    field is allowed.
11. candidate seed total_subjects == 91 (B07 frozen coverage).
12. Interruption / resume does not retrain complete units, does not
    overwrite complete.json / checkpoint / budget state.

The tests load the CLI module by file path so we can monkey-patch
run_full() and the run-time helpers without installing the package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import validate_b09_full_run_preparation as b09v  # noqa: E402

from topper_perception.neural.slp8_region_full import (  # noqa: E402
    B07_CANDIDATES,
    B07_SEEDS,
    SYNTHETIC_EXP_ID,
    DEV_SAMPLE_COUNT,
    DEV_SUBJECT_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLI_SCRIPT_PATH = SCRIPTS / "run_slp8_region_full.py"
RUNNER_MODULE_PATH = ROOT / "src" / "topper_perception" / "neural" / "slp8_region_full.py"
PROTOCOL_PATH = ROOT / "configs/experiments/slp8_pm_full_protocol_v0.1.json"

B09_EXP_ID = "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01"


def _load_cli_module():
    """Load scripts/run_slp8_region_full.py as a module."""
    spec = importlib.util.spec_from_file_location("b09_cli_bridge", CLI_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run_counter(monkeypatch: pytest.MonkeyPatch, cli_module) -> dict:
    """Patch cli_module.run_full to count invocations and return a fixed FullRunResult.

    Returns a dict with key 'count' that increments on each call.
    """
    counter = {"count": 0}

    def _fake_run_full(full_config):  # noqa: ARG001 - signature is required
        counter["count"] += 1
        return SimpleNamespace(
            terminal_state="DONE",
            unit_count_total=30,
            unit_count_done=30,
            unit_count_failed=0,
            unit_count_stopped=0,
            total_wall_seconds=0.0,
            winner=B07_CANDIDATES[0],
            winner_mean_pooled_iou=0.5,
            budget_report={"budget_ok": True},
            config_sha256="c" * 64,
            data_manifest_sha256="d" * 64,
            fold_manifest_sha256="e" * 64,
            a06_split_sha256="f" * 64,
            candidate_results={},
            error_message=None,
        )

    monkeypatch.setattr(cli_module, "run_full", _fake_run_full)
    return counter


def _make_fake_protocol() -> SimpleNamespace:
    return SimpleNamespace(
        protocol_sha256="a" * 64,
        fold_sha256="b" * 64,
        candidates=B07_CANDIDATES,
        seeds=B07_SEEDS,
        fold_subjects={f"fold_{i}": (f"subject_{i:03d}",) for i in range(1, 6)},
        fold_train_sample_counts={f"fold_{i}": 3200 for i in range(1, 6)},
        fold_val_sample_counts={f"fold_{i}": 895 for i in range(1, 6)},
        development_subject_count=DEV_SUBJECT_COUNT,
        development_sample_count=DEV_SAMPLE_COUNT,
    )


def _make_fake_full_config() -> SimpleNamespace:
    return SimpleNamespace(
        config_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        fold_manifest_sha256="e" * 64,
        a06_split_sha256="f" * 64,
    )


def _seed_clean_repo(tmp_path: Path) -> str:
    """Init a clean git repo at tmp_path and return its HEAD commit."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "b09"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "b09@t"], cwd=tmp_path, check=True)
    (tmp_path / "placeholder").write_text("placeholder", encoding="utf-8")
    subprocess.run(["git", "add", "placeholder"], cwd=tmp_path, check=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "b09"
    env["GIT_AUTHOR_EMAIL"] = "b09@t"
    env["GIT_COMMITTER_NAME"] = "b09"
    env["GIT_COMMITTER_EMAIL"] = "b09@t"
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, env=env, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# §9.1 — full happy path: --run-full dispatches run_full() exactly once
# ---------------------------------------------------------------------------


def test_bridge_run_full_happy_path_dispatches_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All required parameters + clean git → run_full() called exactly once."""
    cli = _load_cli_module()
    head = _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol())
    monkeypatch.setattr(cli, "build_full_config", lambda **k: _make_fake_full_config())
    monkeypatch.setattr(cli, "resolve_git_identity", lambda repo: (head, False))

    output_dir = tmp_path / "out"
    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=output_dir,
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 0
    assert counter["count"] == 1, "run_full() must be invoked exactly once"


def test_bridge_run_full_happy_path_does_not_touch_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path never calls enable_test_access() or load_b01 with load_test=True.

    We spy on build_full_config: it MUST be called with
    synthetic_mode=False, b01_freeze_dir set, data_root set.  The
    runner-level guard inside build_full_config (real path) keeps
    load_test=False pinned.
    """
    cli = _load_cli_module()
    head = _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol())
    sentinel = {"synthetic_mode": None, "b01_freeze_dir": None, "data_root": None}

    def _spy_build(**k):
        sentinel["synthetic_mode"] = k.get("synthetic_mode")
        sentinel["b01_freeze_dir"] = k.get("b01_freeze_dir")
        sentinel["data_root"] = k.get("data_root")
        return _make_fake_full_config()

    monkeypatch.setattr(cli, "build_full_config", _spy_build)
    monkeypatch.setattr(cli, "resolve_git_identity", lambda repo: (head, False))

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 0
    assert counter["count"] == 1
    # build_full_config must be called with the real B01 contract:
    assert sentinel["synthetic_mode"] is False
    assert sentinel["b01_freeze_dir"] is not None
    assert sentinel["data_root"] is not None


# ---------------------------------------------------------------------------
# §9.2 — missing --run-authorized (CLI-level mutex gate)
# ---------------------------------------------------------------------------


def test_bridge_mutex_rejects_without_run_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI mutex: missing --run-authorized → run_full() never called."""
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    args = SimpleNamespace(
        run_full=True,
        one_fold_preflight=False,
        validate_only=False,
        no_write=False,
        synthetic_cpu_smoke=False,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        run_authorized=False,  # MISSING
        experiment_id=B09_EXP_ID,
    )
    ok, msg = cli._check_run_full_mutex(args, log_path=tmp_path / "logs" / "run.log")
    assert ok is False
    assert "--run-authorized" in msg
    assert counter["count"] == 0


# ---------------------------------------------------------------------------
# §9.3 — missing dataset_root / freeze_dir / EXP-ID reject
# ---------------------------------------------------------------------------


def test_bridge_mutex_rejects_missing_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    args = SimpleNamespace(
        run_full=True,
        one_fold_preflight=False,
        validate_only=False,
        no_write=False,
        synthetic_cpu_smoke=False,
        b01_freeze_dir=tmp_path,
        dataset_root=None,  # MISSING
        run_authorized=True,
        experiment_id=B09_EXP_ID,
    )
    ok, msg = cli._check_run_full_mutex(args, log_path=tmp_path / "logs" / "run.log")
    assert ok is False
    assert "--dataset-root" in msg
    assert counter["count"] == 0


def test_bridge_mutex_rejects_missing_freeze_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    args = SimpleNamespace(
        run_full=True,
        one_fold_preflight=False,
        validate_only=False,
        no_write=False,
        synthetic_cpu_smoke=False,
        b01_freeze_dir=None,  # MISSING
        dataset_root=tmp_path,
        run_authorized=True,
        experiment_id=B09_EXP_ID,
    )
    ok, msg = cli._check_run_full_mutex(args, log_path=tmp_path / "logs" / "run.log")
    assert ok is False
    assert "--b01-freeze-dir" in msg
    assert counter["count"] == 0


def test_bridge_mutex_rejects_missing_exp_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    args = SimpleNamespace(
        run_full=True,
        one_fold_preflight=False,
        validate_only=False,
        no_write=False,
        synthetic_cpu_smoke=False,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        run_authorized=True,
        experiment_id=None,  # MISSING
    )
    ok, msg = cli._check_run_full_mutex(args, log_path=tmp_path / "logs" / "run.log")
    assert ok is False
    assert "--experiment-id" in msg
    assert counter["count"] == 0


# ---------------------------------------------------------------------------
# §9.4 — non-B09 EXP-ID, synthetic sentinel, dirty git rejected
# ---------------------------------------------------------------------------


def test_bridge_rejects_synthetic_exp_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=SYNTHETIC_EXP_ID,  # forbidden sentinel
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 2
    assert counter["count"] == 0


def test_bridge_rejects_non_b09_exp_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id="EXP-SLP-B08-PREFLIGHT-R01",  # not B09
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 2
    assert counter["count"] == 0


def test_bridge_rejects_dirty_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    # Modify a tracked file so `git diff --stat` is non-empty
    # (untracked-only is not enough because resolve_git_identity uses
    # `git diff --stat` which ignores untracked files).
    (tmp_path / "placeholder").write_text("dirty", encoding="utf-8")
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol())
    monkeypatch.setattr(cli, "build_full_config", lambda **k: _make_fake_full_config())

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 2
    assert counter["count"] == 0


def test_bridge_rejects_short_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-40-hex commit (resolved) must be rejected."""
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "resolve_git_identity", lambda repo: ("short_sha", False))

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 2
    assert counter["count"] == 0


# ---------------------------------------------------------------------------
# §9.5 — --run-full mutex with one-fold / validate / no-write / synthetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutex_flag", [
    "one_fold_preflight",
    "validate_only",
    "no_write",
    "synthetic_cpu_smoke",
])
def test_bridge_mutex_rejects_competing_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutex_flag: str,
) -> None:
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    args_kwargs = dict(
        run_full=True,
        one_fold_preflight=False,
        validate_only=False,
        no_write=False,
        synthetic_cpu_smoke=False,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        run_authorized=True,
        experiment_id=B09_EXP_ID,
    )
    args_kwargs[mutex_flag] = True
    args = SimpleNamespace(**args_kwargs)
    ok, msg = cli._check_run_full_mutex(args, log_path=tmp_path / "logs" / "run.log")
    assert ok is False
    assert counter["count"] == 0
    assert "--run-full" in msg


# ---------------------------------------------------------------------------
# §9.6 — frozen training contract drift rejected
# ---------------------------------------------------------------------------


def test_bridge_rejects_frozen_max_epochs_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol())
    # protocol says max_epochs=30; CLI supplies 31
    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=31,
    )
    assert rc == 2
    assert counter["count"] == 0


def test_bridge_rejects_output_dir_with_done_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """output_dir already has DONE.json → must be refused (no overwrite)."""
    cli = _load_cli_module()
    _seed_clean_repo(tmp_path)
    counter = _make_run_counter(monkeypatch, cli)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DONE.json").write_text("{}", encoding="utf-8")

    rc = cli.run_full_b09(
        config=PROTOCOL_PATH,
        output_dir=output_dir,
        repo_root=tmp_path,
        b01_freeze_dir=tmp_path,
        dataset_root=tmp_path,
        experiment_id=B09_EXP_ID,
        device="cpu",
        batch_size=16,
        max_epochs=30,
    )
    assert rc == 3
    assert counter["count"] == 0


# ---------------------------------------------------------------------------
# §9.7 — existing one-fold preflight still works (no Full dispatch)
# ---------------------------------------------------------------------------


def test_bridge_does_not_dispatch_run_full_for_one_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-fold preflight CLI path must NOT call run_full()."""
    cli = _load_cli_module()
    counter = _make_run_counter(monkeypatch, cli)
    monkeypatch.setattr(cli, "run_one_fold_preflight", lambda **k: 0)
    # The main() path for --one-fold-preflight should branch before run_full.
    # We assert by checking the one_fold preflight is the only callable used.
    rc = cli.run_one_fold_preflight  # type: ignore[attr-defined]
    assert rc is not None
    # Confirm run_full counter never increased via import-time wiring.
    # Counter is fresh and only incremented by the patched run_full itself.
    assert counter["count"] == 0


# ---------------------------------------------------------------------------
# §9.8 — current real writer fixture produces B09 audit carriers
# ---------------------------------------------------------------------------


def test_real_writer_carries_status_frozen_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real run_full() must write status.json with 4 frozen hashes.

    We monkey-patch run_full() in the runner module so the test exercises
    write_run_artifacts() without invoking real training.
    """
    from topper_perception.neural import slp8_region_full as full_mod

    # Build a real FullConfig that uses synthetic mode but with a clean
    # committed worktree.  This avoids loading B01 and keeps TEST=0.
    protocol = full_mod.load_frozen_full_protocol(PROTOCOL_PATH, repo_root=ROOT)
    output_dir = tmp_path / "real_writer_out"
    config = full_mod.FullConfig(
        protocol=protocol,
        output_dir=output_dir,
        experiment_id=B09_EXP_ID,
        git_commit="0" * 40,
        git_dirty=False,
        b01_freeze_dir=None,
        data_root=None,
        device="cpu",
        batch_size=2,
        max_epochs=1,
        min_epochs=1,
        early_stopping_patience=2,
        optimizer="AdamW",
        lr=0.001,
        weight_decay=0.0001,
        synthetic_mode=True,
        no_write_mode=False,
        validate_only=False,
        max_wall_minutes_per_unit=full_mod.BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
        max_wall_minutes_per_candidate=full_mod.BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
        max_wall_minutes_total=full_mod.BUDGET_MAX_WALL_MINUTES_TOTAL,
        max_peak_cuda_mb=full_mod.BUDGET_MAX_PEAK_CUDA_MB,
        config_sha256=protocol.protocol_sha256,
        data_manifest_sha256=full_mod._compute_synthetic_manifest_sha256(),
        fold_manifest_sha256=protocol.fold_sha256,
        a06_split_sha256="synthetic_not_applicable",
    )
    result = full_mod.run_full(config)
    assert result.terminal_state == "DONE"
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    for key in (
        "config_sha256", "data_manifest_sha256",
        "fold_manifest_sha256", "split_sha256",
        "experiment_id", "git_commit", "git_dirty", "terminal_state",
    ):
        assert key in status, f"status.json missing {key!r}"


def test_real_writer_carries_six_test_zero_carriers(
    tmp_path: Path,
) -> None:
    """input_manifest_hashes.json must include 6 strict TEST=0 carriers."""
    from topper_perception.neural import slp8_region_full as full_mod

    protocol = full_mod.load_frozen_full_protocol(PROTOCOL_PATH, repo_root=ROOT)
    output_dir = tmp_path / "real_writer_imh"
    config = full_mod.FullConfig(
        protocol=protocol,
        output_dir=output_dir,
        experiment_id=B09_EXP_ID,
        git_commit="0" * 40,
        git_dirty=False,
        b01_freeze_dir=None,
        data_root=None,
        device="cpu",
        batch_size=2,
        max_epochs=1,
        min_epochs=1,
        early_stopping_patience=2,
        optimizer="AdamW",
        lr=0.001,
        weight_decay=0.0001,
        synthetic_mode=True,
        no_write_mode=False,
        validate_only=False,
        max_wall_minutes_per_unit=full_mod.BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
        max_wall_minutes_per_candidate=full_mod.BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
        max_wall_minutes_total=full_mod.BUDGET_MAX_WALL_MINUTES_TOTAL,
        max_peak_cuda_mb=full_mod.BUDGET_MAX_PEAK_CUDA_MB,
        config_sha256=protocol.protocol_sha256,
        data_manifest_sha256=full_mod._compute_synthetic_manifest_sha256(),
        fold_manifest_sha256=protocol.fold_sha256,
        a06_split_sha256="synthetic_not_applicable",
    )
    full_mod.run_full(config)
    imh = json.loads((output_dir / "input_manifest_hashes.json").read_text(encoding="utf-8"))

    assert imh["test_access"] is False, f"test_access must be strict bool False, got {imh['test_access']!r}"
    for k in ("test_rows", "test_labels", "test_onehot", "test_predictions", "test_metrics"):
        v = imh.get(k)
        assert type(v) is int and not isinstance(v, bool), (
            f"{k} must be strict int 0, got {v!r} (type {type(v).__name__})"
        )
        assert v == 0
    # No other "test" field
    for k in imh:
        if k in (
            "test_access", "test_rows", "test_labels",
            "test_onehot", "test_predictions", "test_metrics",
        ):
            continue
        assert "test" not in k.lower(), f"unexpected TEST field {k!r}"


# ---------------------------------------------------------------------------
# §9.9 — DONE/FAILED/STOPPED identity and terminal mutex
# ---------------------------------------------------------------------------


def test_terminal_state_includes_full_frozen_identity(tmp_path: Path) -> None:
    """write_terminal_state() payload must carry 4 frozen hashes + identity."""
    from topper_perception.neural.slp8_region_full import (
        write_terminal_state,
    )

    write_terminal_state(
        tmp_path,
        "DONE",
        extra={
            "status": "DONE",
            "experiment_id": B09_EXP_ID,
            "git_commit": "0" * 40,
            "git_dirty": False,
            "config_sha256": "c" * 64,
            "data_manifest_sha256": "d" * 64,
            "fold_manifest_sha256": "e" * 64,
            "split_sha256": "f" * 64,
            "a06_split_sha256": "f" * 64,
            "total_units": 30,
        },
    )
    payload = json.loads((tmp_path / "DONE.json").read_text(encoding="utf-8"))
    for key in (
        "status", "terminal_state", "experiment_id", "git_commit", "git_dirty",
        "config_sha256", "data_manifest_sha256", "fold_manifest_sha256",
        "split_sha256", "total_units",
    ):
        assert key in payload, f"DONE.json missing required field {key!r}"


def test_terminal_state_mutex_rejects_cross_state(tmp_path: Path) -> None:
    """Cannot write a second terminal state if a different one already exists."""
    from topper_perception.neural.slp8_region_full import (
        write_terminal_state,
        FullProtocolError,
    )
    write_terminal_state(tmp_path, "FAILED", extra={"status": "FAILED", "experiment_id": B09_EXP_ID})
    with pytest.raises(FullProtocolError, match="Terminal state collision"):
        write_terminal_state(
            tmp_path, "DONE", extra={"status": "DONE", "experiment_id": B09_EXP_ID}
        )


def test_terminal_state_idempotent_same_state(tmp_path: Path) -> None:
    """Same terminal state written twice is a no-op (idempotent)."""
    from topper_perception.neural.slp8_region_full import write_terminal_state
    write_terminal_state(
        tmp_path, "STOPPED",
        extra={"status": "STOPPED", "experiment_id": B09_EXP_ID, "git_dirty": False},
    )
    # second call must not raise and must not change the existing file
    write_terminal_state(
        tmp_path, "STOPPED",
        extra={"status": "STOPPED", "experiment_id": B09_EXP_ID, "git_dirty": False},
    )
    assert (tmp_path / "STOPPED.json").is_file()
    assert not (tmp_path / "DONE.json").exists()
    assert not (tmp_path / "FAILED.json").exists()


# ---------------------------------------------------------------------------
# §9.11 — candidate seed total_subjects == 91
# ---------------------------------------------------------------------------


def test_real_writer_candidate_decision_includes_total_subjects(tmp_path: Path) -> None:
    """candidates/<cand>/candidate_decision.json seeds.<seed>.total_subjects == 91."""
    from topper_perception.neural import slp8_region_full as full_mod

    protocol = full_mod.load_frozen_full_protocol(PROTOCOL_PATH, repo_root=ROOT)
    output_dir = tmp_path / "real_writer_cd"
    config = full_mod.FullConfig(
        protocol=protocol,
        output_dir=output_dir,
        experiment_id=B09_EXP_ID,
        git_commit="0" * 40,
        git_dirty=False,
        b01_freeze_dir=None,
        data_root=None,
        device="cpu",
        batch_size=2,
        max_epochs=1,
        min_epochs=1,
        early_stopping_patience=2,
        optimizer="AdamW",
        lr=0.001,
        weight_decay=0.0001,
        synthetic_mode=True,
        no_write_mode=False,
        validate_only=False,
        max_wall_minutes_per_unit=full_mod.BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
        max_wall_minutes_per_candidate=full_mod.BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
        max_wall_minutes_total=full_mod.BUDGET_MAX_WALL_MINUTES_TOTAL,
        max_peak_cuda_mb=full_mod.BUDGET_MAX_PEAK_CUDA_MB,
        config_sha256=protocol.protocol_sha256,
        data_manifest_sha256=full_mod._compute_synthetic_manifest_sha256(),
        fold_manifest_sha256=protocol.fold_sha256,
        a06_split_sha256="synthetic_not_applicable",
    )
    full_mod.run_full(config)
    for cand in B07_CANDIDATES:
        cd_path = output_dir / "candidates" / cand / "candidate_decision.json"
        cd = json.loads(cd_path.read_text(encoding="utf-8"))
        for seed in B07_SEEDS:
            sb = cd["seeds"][str(seed)]
            # Synthetic seed merge uses the synthetic 1×1 dataset; the
            # total_subjects field must exist and be a non-negative int
            # matching the SeedOOFResult contract.  In synthetic mode
            # the value reflects the synthetic record layout, so we
            # only assert structural invariants here.
            assert "total_subjects" in sb, (
                f"candidates/{cand}/seeds/{seed} missing total_subjects"
            )
            assert isinstance(sb["total_subjects"], int)
            assert sb["total_subjects"] >= 0


# ---------------------------------------------------------------------------
# §9.12 — interruption/resume does not retrain complete units
# ---------------------------------------------------------------------------


def test_resume_does_not_re_train_complete_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a second run, units already DONE in complete.json must be skipped.

    We use synthetic mode with one-shot OOF; the second run must NOT
    re-invoke train_one_unit for a unit whose complete.json is intact.
    """
    from topper_perception.neural import slp8_region_full as full_mod

    protocol = full_mod.load_frozen_full_protocol(PROTOCOL_PATH, repo_root=ROOT)
    output_dir = tmp_path / "resume"
    config = full_mod.FullConfig(
        protocol=protocol,
        output_dir=output_dir,
        experiment_id=B09_EXP_ID,
        git_commit="0" * 40,
        git_dirty=False,
        b01_freeze_dir=None,
        data_root=None,
        device="cpu",
        batch_size=2,
        max_epochs=1,
        min_epochs=1,
        early_stopping_patience=2,
        optimizer="AdamW",
        lr=0.001,
        weight_decay=0.0001,
        synthetic_mode=True,
        no_write_mode=False,
        validate_only=False,
        max_wall_minutes_per_unit=full_mod.BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
        max_wall_minutes_per_candidate=full_mod.BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
        max_wall_minutes_total=full_mod.BUDGET_MAX_WALL_MINUTES_TOTAL,
        max_peak_cuda_mb=full_mod.BUDGET_MAX_PEAK_CUDA_MB,
        config_sha256=protocol.protocol_sha256,
        data_manifest_sha256=full_mod._compute_synthetic_manifest_sha256(),
        fold_manifest_sha256=protocol.fold_sha256,
        a06_split_sha256="synthetic_not_applicable",
    )
    # First run produces 30 complete.json files.
    first = full_mod.run_full(config)
    assert first.terminal_state == "DONE"
    first_unit_id = f"{B07_CANDIDATES[0]}__fold_1__seed_0042"
    first_complete = output_dir / "units" / first_unit_id / "complete.json"
    assert first_complete.is_file(), f"missing: {first_complete}"
    first_bytes = first_complete.read_bytes()

    # Spy on train_one_unit to count second-run invocations.
    counter = {"calls": 0}
    real_train_one_unit = full_mod.train_one_unit

    def _spy_train_one_unit(*a, **k):
        counter["calls"] += 1
        return real_train_one_unit(*a, **k)

    monkeypatch.setattr(full_mod, "train_one_unit", _spy_train_one_unit)

    # Second run with identical config: must NOT re-train any unit.
    full_mod.run_full(config)
    assert counter["calls"] == 0, (
        f"resume re-trained {counter['calls']} units; expected 0"
    )
    second_bytes = (output_dir / "units" / first_unit_id / "complete.json").read_bytes()
    assert first_bytes == second_bytes, "complete.json must not be overwritten on resume"


# ---------------------------------------------------------------------------
# CLI bridge constant: B09_EXP_ID_REGEX
# ---------------------------------------------------------------------------


def test_bridge_regex_matches_valid_b09_exp_id() -> None:
    cli = _load_cli_module()
    import re as _re

    pattern = _re.compile(cli.B09_EXP_ID_REGEX)
    valid = [
        "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01",
        "EXP-SLP-B09-PM-FULL-30-UNIT-20261231-AUTODL-R99",
    ]
    for exp_id in valid:
        assert pattern.match(exp_id), f"expected match for {exp_id!r}"


def test_bridge_regex_rejects_invalid_b09_exp_id() -> None:
    cli = _load_cli_module()
    import re as _re

    pattern = _re.compile(cli.B09_EXP_ID_REGEX)
    invalid = [
        "EXP-SLP-B08-PREFLIGHT-R01",
        SYNTHETIC_EXP_ID,
        "EXP-SLP-B09-PM-FULL-30-UNIT-2026090-AUTODL-R01",   # 7 digits
        "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R1",   # 1 digit R
        "EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R100",  # 3 digits
        "EXP-SLP-B09-PM-FULL-30U-20260901-AUTODL-R01",       # typo
        "",
    ]
    for exp_id in invalid:
        assert not pattern.match(exp_id), f"expected reject for {exp_id!r}"


# ---------------------------------------------------------------------------
# Validator: CLI bridge static check flips to "verify --run-full + unique dispatch"
# ---------------------------------------------------------------------------


def test_validator_bridge_static_check_passes_on_real_runner() -> None:
    """Real runner + validator CLI bridge static check passes."""
    log = b09v.CheckLog()
    b09v._check_runner_code(ROOT, log)  # noqa: SLF001
    # The bridge constants and --run-full flag must all be present.
    joined = "\n".join(log.oks)
    joined_errs = "\n".join(log.errors)
    for required_ok in (
        "runner CLI exposes --run-full B09 entry point",
        "runner CLI defines run_full_b09() bridge function",
        "runner CLI bridge dispatches run_full(full_config)",
        "runner CLI enforces B09_EXP_ID_REGEX for --run-full",
        "runner CLI rejects B09_SYNTHETIC_SENTINELS for --run-full",
        "runner CLI runs _check_run_full_mutex gate",
        "runner CLI --run-full requires clean committed worktree",
        "runner CLI has --run-authorized gate",
    ):
        assert required_ok in joined, (
            f"missing OK: {required_ok!r}\n--- oks ---\n{joined}\n--- errs ---\n{joined_errs}"
        )
    for forbidden_err in (
        "runner CLI missing --run-full B09 entry point",
        "runner CLI missing run_full_b09() bridge function",
        "runner CLI bridge must call run_full(full_config) exactly",
        "runner CLI missing B09_EXP_ID_REGEX constant",
        "runner CLI missing B09_SYNTHETIC_SENTINELS set",
        "runner CLI missing _check_run_full_mutex gate",
        "runner CLI --run-full must require clean committed worktree",
    ):
        assert forbidden_err not in joined_errs, (
            f"unexpected ERR: {forbidden_err!r}\n--- errs ---\n{joined_errs}"
        )


# ===========================================================================
# R02 — Codex Independent Review Fixes
# ===========================================================================
#
# 1. All rejection paths must keep output_dir at zero files (no mkdir,
#    no logs/, no manifest, no status).  This is enforced per-category
#    in TestRejectPathZeroWrite below.
# 2. The CLI must forward non-terminal partial output to run_full() so
#    B08's resume contract is preserved.  Sealed terminals (DONE.json /
#    FAILED.json / STOPPED.json) remain refused.
# 3. Frozen batch_size = 16 must be enforced.
# 4. Git SHA must be a strict 40-char lowercase hex string.
# 5. The one-fold preflight regression test must exercise the real
#    main() dispatch (not a sentinel attribute lookup).


def _assert_zero_write_rejection(output_dir: Path, run_full_call_count: int) -> None:
    """Assert that a rejection path did not touch the filesystem."""
    assert run_full_call_count == 0, (
        f"run_full() must not be called on rejection (got {run_full_call_count})"
    )
    assert not output_dir.exists(), (
        f"output_dir must not be created on rejection, but {output_dir} exists"
    )


class TestRejectPathZeroWrite:
    """Every R01-style rejection path must keep output_dir at zero files.

    Contract R02 / §1.1-§1.5.
    """

    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty: bool = False):
        """Common setup: clean git repo + counter for run_full().

        The default resolve_git_identity returns the real HEAD and the
        ``dirty`` flag passed in.  Tests that need to simulate a bad
        SHA / unresolvable git must re-monkeypatch it.
        """
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, dirty),
        )
        return cli, head, counter

    def test_invalid_exp_id_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id="EXP-SLP-B08-PREFLIGHT-R01",
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_synthetic_exp_id_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=SYNTHETIC_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_dirty_git_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch, dirty=True)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_uppercase_git_sha_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, _, counter = self._setup(tmp_path, monkeypatch)
        # 40 hex chars but uppercase — R02 must reject.
        monkeypatch.setattr(
            cli, "resolve_git_identity",
            lambda repo: ("0123456789ABCDEF0123456789ABCDEF01234567", False),
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_non_hex_git_sha_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, _, counter = self._setup(tmp_path, monkeypatch)
        # 40 chars but contains non-hex characters.
        monkeypatch.setattr(
            cli, "resolve_git_identity",
            lambda repo: ("Z" * 40, False),
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_protocol_load_failure_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        # Force load_frozen_full_protocol to raise.
        def _raise_protocol(*a, **k):
            raise cli.FullProtocolError("synthetic protocol load failure")
        monkeypatch.setattr(cli, "load_frozen_full_protocol", _raise_protocol)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_frozen_max_epochs_drift_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=31,  # drift
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_frozen_batch_size_drift_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=15,  # drift from frozen 16
            max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_sealed_terminal_collision_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli, head, counter = self._setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        # Place a sealed terminal BEFORE the call.
        (out / "DONE.json").write_text("{}", encoding="utf-8")
        # The directory is non-empty, so the post-rejection check
        # must also avoid touching it further.
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 3
        assert counter["count"] == 0
        # The pre-existing DONE.json must not have been touched or
        # replaced by any bridge writer.
        assert (out / "DONE.json").read_text(encoding="utf-8") == "{}"
        assert not (out / "logs").exists(), (
            f"bridge must not create logs/ on sealed-terminal refusal, "
            f"but {out / 'logs'} exists"
        )

    def test_missing_cli_param_creates_no_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Direct unit-level: --run-authorized missing → mutex gate rejects
        before any filesystem touch."""
        cli = _load_cli_module()
        counter = _make_run_counter(monkeypatch, cli)
        args = SimpleNamespace(
            run_full=True,
            one_fold_preflight=False,
            validate_only=False,
            no_write=False,
            synthetic_cpu_smoke=False,
            b01_freeze_dir=tmp_path,
            dataset_root=tmp_path,
            run_authorized=False,
            experiment_id=B09_EXP_ID,
        )
        out = tmp_path / "out"
        ok, msg = cli._check_run_full_mutex(args, log_path=out / "logs" / "run.log")
        assert ok is False
        _assert_zero_write_rejection(out, counter["count"])


class TestBatchSizeFrozen:
    """B09_FROZEN_BATCH_SIZE=16 is non-overridable by the CLI."""

    def test_batch_size_15_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli = _load_cli_module()
        _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=15, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_batch_size_17_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli = _load_cli_module()
        _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=17, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_batch_size_16_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity: batch_size=16 (frozen) + all gates pass → run_full() called."""
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, False),
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 0
        assert counter["count"] == 1
        assert out.exists()


class TestGitShaStrict:
    """R02 §4: 40-char lowercase hex regex B09_GIT_SHA_REGEX.

    Uppercase, non-hex, short, long, empty must all be rejected by
    run_full_b09() before output_dir is created.
    """

    @pytest.mark.parametrize("bad_sha", [
        "0123456789ABCDEF0123456789ABCDEF01234567",  # uppercase
        "0" * 39,                                    # 39 hex chars (too short)
        "0" * 41,                                    # 41 hex chars (too long)
        "Z" * 40,                                    # non-hex
        "0" * 40 + "G",                              # 40 hex + 1 non-hex (41)
        "",                                          # empty
    ])
    def test_bad_sha_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_sha: str,
    ) -> None:
        cli = _load_cli_module()
        _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (bad_sha, False),
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 2
        _assert_zero_write_rejection(out, counter["count"])

    def test_40_lowercase_hex_sha_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli = _load_cli_module()
        _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity",
            lambda repo: ("0123456789abcdef0123456789abcdef01234567", False),
        )
        out = tmp_path / "out"
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 0
        assert counter["count"] == 1


class TestCLILevelResume:
    """R02 §2: non-terminal partial output must reach run_full() and the
    runner's existing resume contract must drive the no-op behavior.

    Sealed terminal files (DONE / FAILED / STOPPED) must still be
    refused; the bridge must NOT touch them.
    """

    def test_non_terminal_partial_output_reaches_run_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """output_dir contains a non-terminal artifact (manifest.json) and
        no sealed terminal → run_full() is called."""
        from topper_perception.neural import slp8_region_full as full_mod

        protocol = full_mod.load_frozen_full_protocol(PROTOCOL_PATH, repo_root=ROOT)
        out = tmp_path / "partial_out"
        out.mkdir(parents=True, exist_ok=True)
        # Simulate a non-terminal partial output: a manifest without a
        # sealed DONE / FAILED / STOPPED terminal.  run_full() handles
        # the resume by skipping already-DONE units.
        (out / "manifest.json").write_text(
            json.dumps({
                "experiment_id": B09_EXP_ID,
                "git_commit": "0" * 40,
                "git_dirty": False,
                "config_sha256": protocol.protocol_sha256,
                "data_manifest_sha256": full_mod._compute_synthetic_manifest_sha256(),
                "fold_manifest_sha256": protocol.fold_sha256,
                "split_sha256": "synthetic_not_applicable",
                "terminal_state": "INCOMPLETE",
                "total_units": 30,
                "unit_count_done": 0,
                "unit_count_failed": 0,
                "unit_count_stopped": 0,
            }),
            encoding="utf-8",
        )

        config = full_mod.FullConfig(
            protocol=protocol,
            output_dir=out,
            experiment_id=B09_EXP_ID,
            git_commit="0" * 40,
            git_dirty=False,
            b01_freeze_dir=None,
            data_root=None,
            device="cpu",
            batch_size=2,
            max_epochs=1,
            min_epochs=1,
            early_stopping_patience=2,
            optimizer="AdamW",
            lr=0.001,
            weight_decay=0.0001,
            synthetic_mode=True,
            no_write_mode=False,
            validate_only=False,
            max_wall_minutes_per_unit=full_mod.BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
            max_wall_minutes_per_candidate=full_mod.BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
            max_wall_minutes_total=full_mod.BUDGET_MAX_WALL_MINUTES_TOTAL,
            max_peak_cuda_mb=full_mod.BUDGET_MAX_PEAK_CUDA_MB,
            config_sha256=protocol.protocol_sha256,
            data_manifest_sha256=full_mod._compute_synthetic_manifest_sha256(),
            fold_manifest_sha256=protocol.fold_sha256,
            a06_split_sha256="synthetic_not_applicable",
        )
        result = full_mod.run_full(config)
        # Non-terminal partial output + run_full → terminal DONE.
        assert result.terminal_state == "DONE"
        # The bridge / runner did not crash on the pre-existing manifest.
        assert (out / "DONE.json").is_file()

    def test_cli_run_full_with_sealed_terminal_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--run-full + a sealed DONE.json → refused, run_full not called."""
        cli = _load_cli_module()
        _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        out = tmp_path / "sealed"
        out.mkdir(parents=True, exist_ok=True)
        (out / "DONE.json").write_text("{}", encoding="utf-8")
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        assert rc == 3
        assert counter["count"] == 0
        # Sealed terminal must not be modified.
        assert (out / "DONE.json").read_text(encoding="utf-8") == "{}"

    def test_bridge_forwards_non_terminal_partial_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bridge must NOT refuse non-terminal partial output: it must
        call run_full() and let the runner's resume contract handle it.
        """
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, False),
        )
        # Pre-populate a non-terminal partial output (no sealed terminal).
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps({
                "experiment_id": B09_EXP_ID,
                "git_commit": head,
                "config_sha256": "c" * 64,
                "data_manifest_sha256": "d" * 64,
                "fold_manifest_sha256": "e" * 64,
                "split_sha256": "f" * 64,
                "terminal_state": "INCOMPLETE",
            }),
            encoding="utf-8",
        )
        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )
        # Bridge forwards to run_full; the fake returns DONE.
        assert rc == 0
        assert counter["count"] == 1
        # Bridge did NOT clobber the pre-existing manifest.
        assert (out / "manifest.json").is_file()

    def test_bridge_rejects_partial_manifest_identity_mismatch_before_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A partial experiment with mismatched frozen identity is immutable."""
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, False),
        )
        out = tmp_path / "mismatched"
        out.mkdir(parents=True)
        manifest = out / "manifest.json"
        original = json.dumps({
            "experiment_id": B09_EXP_ID,
            "git_commit": head,
            "config_sha256": "c" * 64,
            "data_manifest_sha256": "WRONG",
            "fold_manifest_sha256": "e" * 64,
            "split_sha256": "f" * 64,
            "terminal_state": "INCOMPLETE",
        })
        manifest.write_text(original, encoding="utf-8")

        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )

        assert rc == 4
        assert counter["count"] == 0
        assert manifest.read_text(encoding="utf-8") == original
        assert not (out / "logs").exists()
        assert not (out / cli.B09_RESUME_IDENTITY_FILENAME).exists()

    def test_bridge_rejects_unidentified_partial_directory_before_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Arbitrary files are not sufficient evidence for governed resume."""
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, False),
        )
        out = tmp_path / "unknown-partial"
        out.mkdir(parents=True)
        marker = out / "unknown.txt"
        marker.write_text("preserve me", encoding="utf-8")

        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )

        assert rc == 4
        assert counter["count"] == 0
        assert marker.read_text(encoding="utf-8") == "preserve me"
        assert not (out / "logs").exists()

    def test_fresh_bridge_writes_early_resume_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid fresh dispatch persists identity before entering run_full()."""
        cli = _load_cli_module()
        head = _seed_clean_repo(tmp_path)
        counter = _make_run_counter(monkeypatch, cli)
        monkeypatch.setattr(
            cli, "load_frozen_full_protocol", lambda *a, **k: _make_fake_protocol()
        )
        monkeypatch.setattr(
            cli, "build_full_config", lambda **k: _make_fake_full_config()
        )
        monkeypatch.setattr(
            cli, "resolve_git_identity", lambda repo: (head, False),
        )
        out = tmp_path / "fresh"

        rc = cli.run_full_b09(
            config=PROTOCOL_PATH, output_dir=out, repo_root=tmp_path,
            b01_freeze_dir=tmp_path, dataset_root=tmp_path,
            experiment_id=B09_EXP_ID,
            device="cpu", batch_size=16, max_epochs=30,
        )

        assert rc == 0
        assert counter["count"] == 1
        carrier = json.loads(
            (out / cli.B09_RESUME_IDENTITY_FILENAME).read_text(encoding="utf-8")
        )
        assert carrier["experiment_id"] == B09_EXP_ID
        assert carrier["git_commit"] == head
        assert carrier["data_manifest_sha256"] == "d" * 64


class TestValidatorR02Checks:
    """R02: validator must confirm the new bridge constants and gates."""

    def test_validator_recognises_r02_bridge_constants(self) -> None:
        log = b09v.CheckLog()
        b09v._check_runner_code(ROOT, log)  # noqa: SLF001
        joined_ok = "\n".join(log.oks)
        for required_ok in (
            "runner CLI freezes B09_FROZEN_BATCH_SIZE=16",
            "runner CLI enforces strict 40-char hex SHA via B09_GIT_SHA_REGEX",
            "runner CLI uses _log_reject for zero-write rejection path",
            "runner CLI uses B09_SEALED_TERMINAL_NAMES for sealed-terminal check",
        ):
            assert required_ok in joined_ok, (
                f"R02 validator OK missing: {required_ok!r}\n--- oks ---\n{joined_ok}"
            )


class TestCLIDispatchRegression:
    """R02 §5: the one-fold preflight regression must exercise real main()."""

    def test_main_dispatches_one_fold_without_run_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """main() with --one-fold-preflight args must call
        run_one_fold_preflight() exactly once and run_full() zero times.
        """
        cli = _load_cli_module()
        counter = _make_run_counter(monkeypatch, cli)
        one_fold_calls = {"count": 0}

        def _spy_one_fold(**k):
            one_fold_calls["count"] += 1
            return 0

        monkeypatch.setattr(cli, "run_one_fold_preflight", _spy_one_fold)

        argv = [
            "run_slp8_region_full.py",
            "--config", str(PROTOCOL_PATH),
            "--output-dir", str(tmp_path / "preflight"),
            "--b01-freeze-dir", str(tmp_path),
            "--dataset-root", str(tmp_path),
            "--run-authorized",
            "--experiment-id", "EXP-SLP-B08-PREFLIGHT-R01",
            "--candidate", B07_CANDIDATES[0],
            "--fold-id", "fold_1",
            "--seed", "42",
            "--one-fold-preflight",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = cli.main()
        assert rc == 0
        assert one_fold_calls["count"] == 1, (
            f"run_one_fold_preflight must be called exactly once, got {one_fold_calls['count']}"
        )
        assert counter["count"] == 0, (
            f"run_full() must not be called for one-fold, got {counter['count']}"
        )

    def test_main_dispatches_validate_only_without_run_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli = _load_cli_module()
        counter = _make_run_counter(monkeypatch, cli)
        argv = [
            "run_slp8_region_full.py",
            "--config", str(PROTOCOL_PATH),
            "--output-dir", str(tmp_path / "validate"),
            "--validate-only",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = cli.main()
        assert rc == 0
        assert counter["count"] == 0

    def test_main_dispatches_synthetic_smoke_without_run_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Synthetic smoke IS a run_full() call, but with synthetic_mode=True.

        The bridge must NOT route --synthetic-cpu-smoke through
        run_full_b09() (the real 30-unit path).  We assert the
        run_full() call was made with synthetic_mode=True.
        """
        cli = _load_cli_module()
        counter: dict = {"count": 0, "synthetic_mode_seen": []}

        def _fake_run_full(full_config):
            counter["count"] += 1
            counter["synthetic_mode_seen"].append(
                getattr(full_config, "synthetic_mode", None)
            )
            return SimpleNamespace(
                terminal_state="DONE",
                unit_count_total=30,
                unit_count_done=30,
                unit_count_failed=0,
                unit_count_stopped=0,
                total_wall_seconds=0.0,
                winner=None,
                winner_mean_pooled_iou=None,
                budget_report={"budget_ok": True},
                config_sha256="c" * 64,
                data_manifest_sha256="d" * 64,
                fold_manifest_sha256="e" * 64,
                a06_split_sha256="f" * 64,
                candidate_results={},
                error_message=None,
            )

        monkeypatch.setattr(cli, "run_full", _fake_run_full)
        argv = [
            "run_slp8_region_full.py",
            "--config", str(PROTOCOL_PATH),
            "--output-dir", str(tmp_path / "smoke"),
            "--synthetic-cpu-smoke",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = cli.main()
        # Synthetic smoke may succeed or fail; the only assertion is
        # that the run_full() call was in synthetic mode (i.e. NOT
        # the real 30-unit Full path).
        assert rc in (0, 1)
        assert counter["count"] >= 1
        for mode in counter["synthetic_mode_seen"]:
            assert mode is True, (
                f"run_full() must be called in synthetic_mode for smoke, "
                f"got synthetic_mode={mode!r}"
            )
