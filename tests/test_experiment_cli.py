"""CLI exit-code and artifact tests for the experiment runner entry point."""

from __future__ import annotations

import json
from pathlib import Path

from topper_perception.experiments import runner as runner_module
from topper_perception.experiments.runner import main


def _write_config(path: Path, **overrides) -> Path:
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": "EXP-CLI-SMOKE-20260819-R01",
        "task_id": "TASK-EXP-RUNNER-B-v0.1",
        "scope": "smoke",
        "runner_type": "dummy",
        "seed": 42,
        "output_root": "outputs/experiments",
        "parameters": {"n_samples": 256},
    }
    config.update(overrides)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cli_success_exits_zero(tmp_path: Path, capsys) -> None:
    cfg = _write_config(tmp_path / "ok.json")
    out = tmp_path / "out"
    code = main(["--config", str(cfg), "--output-root", str(out)])
    assert code == 0
    assert (out / "EXP-CLI-SMOKE-20260819-R01" / "DONE.json").is_file()
    assert "SUCCEEDED" in capsys.readouterr().out


def test_cli_invalid_config_exits_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path / "bad.json", scope="huge")
    code = main(["--config", str(cfg), "--output-root", str(tmp_path / "out")])
    assert code == 2


def test_cli_dirty_gate_exits_two(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "capture_git_info",
        lambda _root: {"repo": True, "sha": None, "branch": None, "dirty": None},
    )
    cfg = _write_config(
        tmp_path / "mini.json",
        exp_id="EXP-CLI-MINI-20260819-R01",
        scope="mini",
    )
    code = main(["--config", str(cfg), "--output-root", str(tmp_path / "out")])
    assert code == 2


def test_cli_exp_id_conflict_exits_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path / "ok.json")
    out = tmp_path / "out"
    assert main(["--config", str(cfg), "--output-root", str(out)]) == 0
    # Same EXP-ID + same output root must refuse to overwrite.
    assert main(["--config", str(cfg), "--output-root", str(out)]) == 2


def test_cli_runner_failure_exits_one(tmp_path: Path) -> None:
    cfg = _write_config(
        tmp_path / "fail.json",
        parameters={"n_samples": 256, "fail": True},
    )
    out = tmp_path / "out"
    code = main(["--config", str(cfg), "--output-root", str(out)])
    assert code == 1
    assert (out / "EXP-CLI-SMOKE-20260819-R01" / "FAILED.json").is_file()
