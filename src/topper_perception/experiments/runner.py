"""Minimal, dataset/model-agnostic experiment runner.

Turns a validated experiment config into a governed run: resolve the config,
enforce the QUEUED gate (clean worktree for mini/full), create the per-EXP-ID
artifact directory, execute the registered runner, and write status/manifest/
metrics/DONE (or FAILED) artifacts — without touching PoPu/SLP/PressurePose and
without any deep-learning dependency.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from topper_perception.experiments.artifacts import (
    atomic_write_json,
    capture_git_info,
    capture_system_info,
    compute_config_hash,
)
from topper_perception.experiments.contracts import (
    ConfigValidationError,
    DirtyWorktreeError,
    ExpIdError,
    ExperimentConfig,
    State,
    StateTransitionError,
    transition,
    validate_experiment_config,
)

GitInfoProvider = Callable[[Path], dict[str, Any]]
SystemInfoProvider = Callable[[], dict[str, Any]]
RunnerFn = Callable[[Mapping[str, Any], int], dict[str, Any]]


@dataclass(frozen=True)
class RunResult:
    """Outcome of a successful governed run."""

    exp_id: str
    experiment_dir: Path
    state: str
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _project_root() -> Path:
    """Locate the repository root by walking up to ``pyproject.toml``."""
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here


PROJECT_ROOT = _project_root()


def _project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dummy_smoke(parameters: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Deterministic dummy runner: same config+seed -> same metrics.

    Supports a deliberate failure via ``parameters.fail = true`` for testing the
    failure path. Does not read PoPu/SLP/PressurePose.
    """
    params = dict(parameters)
    if bool(params.get("fail", False)):
        raise RuntimeError("Dummy smoke configured to fail (parameters.fail=true).")

    n_samples = int(params.get("n_samples", 100))
    if n_samples <= 0:
        raise ValueError("parameters.n_samples must be a positive integer.")

    rng = random.Random(seed)
    accuracy = round(0.80 + 0.19 * rng.random(), 6)
    balanced_accuracy = round(accuracy - 0.005 + 0.01 * rng.random(), 6)
    macro_f1 = round(0.78 + 0.20 * rng.random(), 6)
    loss = round(0.10 + 0.40 * rng.random(), 6)
    return {
        "n_samples": n_samples,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "loss": loss,
    }


#: Registry mapping ``runner_type`` to its execution function.
RUNNER_REGISTRY: dict[str, RunnerFn] = {
    "dummy": run_dummy_smoke,
}


def _resolved_config(
    cfg: ExperimentConfig,
    output_root: Path,
    experiment_dir: Path,
    git_info: Mapping[str, Any],
    execution_command: str,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": cfg.schema_version,
        "exp_id": cfg.exp_id,
        "task_id": cfg.task_id,
        "scope": cfg.scope,
        "runner_type": cfg.runner_type,
        "seed": cfg.seed,
        "output_root": str(output_root),
        "experiment_dir": str(experiment_dir),
        "parameters": dict(cfg.parameters),
        "git": dict(git_info),
        "execution_command": execution_command,
        "config_hash": config_hash,
        "resolved_at_utc": _utcnow(),
    }


def _write_status(
    experiment_dir: Path, state: State, history: list[dict[str, Any]]
) -> None:
    atomic_write_json(
        experiment_dir / "status.json",
        {
            "exp_id": experiment_dir.name,
            "state": state,
            "updated_at_utc": _utcnow(),
            "history": history,
        },
    )


def _write_run_log(run_log: Path, text: str) -> None:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as handle:
        handle.write(text)


def run_experiment(
    config: Mapping[str, Any],
    *,
    output_root: Path | str | None = None,
    execution_command: str = "",
    project_root: Path | None = None,
    git_info_provider: GitInfoProvider | None = None,
    system_info_provider: SystemInfoProvider | None = None,
) -> RunResult:
    """Run a governed experiment from a raw config mapping.

    Raises a specific :class:`ExperimentError` subclass on validation, EXP-ID,
    gating, or state-machine failures; re-raises execution exceptions after
    recording a truthful ``FAILED.json`` (never swallowing the original cause).
    """
    cfg = validate_experiment_config(config)

    project_root = project_root or PROJECT_ROOT
    root = Path(output_root) if output_root is not None else Path(cfg.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    experiment_dir = root / cfg.exp_id

    if experiment_dir.exists():
        raise ExpIdError(
            f"EXP-ID {cfg.exp_id!r} already exists at {experiment_dir}; "
            "refusing to overwrite an existing experiment."
        )

    git_provider = git_info_provider or capture_git_info
    git_info = git_provider(project_root)

    if cfg.scope in ("mini", "full") and git_info.get("dirty"):
        raise DirtyWorktreeError(
            f"scope={cfg.scope!r} requires a clean Git worktree, but the "
            f"worktree is dirty (sha={git_info.get('sha')!r})."
        )

    system_info = (system_info_provider or capture_system_info)()

    config_hash = compute_config_hash(cfg.raw)
    resolved = _resolved_config(
        cfg, root, experiment_dir, git_info, execution_command, config_hash
    )

    experiment_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / "run.log"

    history: list[dict[str, Any]] = []

    def _advance(prev: State | None, nxt: State) -> State:
        if prev is not None:
            transition(prev, nxt)
        history.append({"from": prev, "to": nxt, "at_utc": _utcnow()})
        _write_status(experiment_dir, nxt, history)
        return nxt

    _write_run_log(run_log, f"resolved config hash: {config_hash}\n")

    manifest: dict[str, Any] = {
        "exp_id": cfg.exp_id,
        "task_id": cfg.task_id,
        "schema_version": cfg.schema_version,
        "scope": cfg.scope,
        "runner_type": cfg.runner_type,
        "git": dict(git_info),
        "config_hash": config_hash,
        "python_version": system_info.get("python_version"),
        "os": system_info.get("os"),
        "cpu": system_info.get("cpu"),
        "gpu": system_info.get("gpu"),
        "cuda": system_info.get("cuda"),
        "seed": cfg.seed,
        "execution_command": execution_command,
        "started_at_utc": _utcnow(),
        "ended_at_utc": None,
    }

    atomic_write_json(experiment_dir / "resolved_config.json", resolved)
    atomic_write_json(experiment_dir / "manifest.json", manifest)

    current_state: State | None = _advance(None, State.QUEUED)

    try:
        current_state = _advance(current_state, State.RUNNING)

        runner_fn = RUNNER_REGISTRY[cfg.runner_type]
        metrics = runner_fn(cfg.parameters, cfg.seed)

        atomic_write_json(experiment_dir / "metrics.json", metrics)
        _write_run_log(run_log, f"metrics: {json.dumps(metrics, ensure_ascii=False)}\n")

        manifest["ended_at_utc"] = _utcnow()
        atomic_write_json(experiment_dir / "manifest.json", manifest)

        current_state = _advance(current_state, State.SUCCEEDED)
        atomic_write_json(
            experiment_dir / "DONE.json",
            {"status": "SUCCEEDED", "exp_id": cfg.exp_id},
        )
    except BaseException as exc:
        manifest["ended_at_utc"] = _utcnow()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(experiment_dir / "manifest.json", manifest)
        _write_run_log(run_log, f"ERROR:\n{traceback.format_exc()}")
        try:
            _advance(current_state, State.FAILED)
        except StateTransitionError:
            pass
        atomic_write_json(
            experiment_dir / "FAILED.json",
            {
                "status": "FAILED",
                "exp_id": cfg.exp_id,
                "error": f"{type(exc).__name__}: {exc}",
                "ended_at_utc": _utcnow(),
            },
        )
        raise

    return RunResult(
        exp_id=cfg.exp_id,
        experiment_dir=experiment_dir,
        state=State.SUCCEEDED,
        metrics=metrics,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/runner_smoke_v0.1.json"),
        help="Experiment config JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the config's output_root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_json(_project_path(args.config))
    try:
        result = run_experiment(
            config,
            output_root=args.output_root,
            execution_command=" ".join(sys.argv),
        )
    except (ConfigValidationError, ExpIdError, DirtyWorktreeError, StateTransitionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except BaseException:
        traceback.print_exc()
        return 1
    print(
        f"SUCCEEDED exp_id={result.exp_id} dir={result.experiment_dir} "
        f"metrics={json.dumps(result.metrics, ensure_ascii=False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
