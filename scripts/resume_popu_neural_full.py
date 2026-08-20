"""Resume/finalize a governed P5.2-C Full run without retraining completed units."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from topper_perception.experiments.artifacts import (
    atomic_write_json,
    capture_git_info,
    compute_config_hash,
)
from topper_perception.experiments.contracts import DirtyWorktreeError, validate_experiment_config
from topper_perception.neural.full import PROJECT_ROOT, run_popu_neural_full
from topper_perception.neural.full_protocol import validate_full_config


DEFAULT_CONFIG = Path("configs/experiments/popu_neural_full_v0.1.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _status(exp_dir: Path, state: str, history: list[dict]) -> None:
    atomic_write_json(
        exp_dir / "status.json",
        {
            "exp_id": exp_dir.name,
            "state": state,
            "updated_at_utc": _utcnow(),
            "history": history,
        },
    )


def _append_log(exp_dir: Path, message: str) -> None:
    path = exp_dir / "logs" / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = validate_experiment_config(config)
    validate_full_config(config)
    if cfg.scope != "full" or cfg.runner_type != "popu_neural_full":
        raise ValueError("Resume command accepts only the frozen P5.2-C Full config.")

    output_root = _project_path(Path(cfg.output_root))
    exp_dir = output_root / cfg.exp_id
    if not exp_dir.is_dir() or (exp_dir / "DONE.json").exists():
        raise FileNotFoundError("Expected an incomplete existing Full experiment directory.")

    resolved = json.loads((exp_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if resolved.get("config_hash") != compute_config_hash(config):
        raise ValueError("Existing Full experiment config hash differs from the frozen config.")
    status = json.loads((exp_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") not in {"FAILED", "RUNNING"}:
        raise ValueError(f"Cannot resume experiment in state {status.get('state')!r}.")

    git = capture_git_info(PROJECT_ROOT)
    if not (
        git.get("repo") is True
        and isinstance(git.get("sha"), str)
        and bool(git.get("sha"))
        and git.get("dirty") is False
    ):
        raise DirtyWorktreeError(f"Resume requires a clean committed worktree, got {git!r}.")

    history = list(status.get("history", []))
    previous_state = str(status["state"])
    previous_failed = exp_dir / "FAILED.json"
    if previous_failed.is_file():
        archived = exp_dir / "logs" / f"FAILED.before-resume.{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(previous_failed), str(archived))

    resume_event = {
        "at_utc": _utcnow(),
        "reason": "finalize completed fold transactions after aggregation-only fix",
        "from_git": resolved.get("git", {}).get("sha"),
        "resume_git": git["sha"],
        "completed_units_before_resume": len(list(exp_dir.glob("folds/repeat_*/fold_*/*/complete.json"))),
    }
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("resume_events", []).append(resume_event)
    manifest["ended_at_utc"] = None
    atomic_write_json(manifest_path, manifest)
    history.append(
        {"from": previous_state, "to": "RUNNING", "at_utc": _utcnow(), "resume": True}
    )
    _status(exp_dir, "RUNNING", history)
    _append_log(exp_dir, f"RESUME: {json.dumps(resume_event, ensure_ascii=False)}\n")

    try:
        metrics = run_popu_neural_full(cfg.parameters, cfg.seed, exp_dir)
        atomic_write_json(exp_dir / "metrics.json", metrics)
        manifest["ended_at_utc"] = _utcnow()
        manifest.pop("error", None)
        atomic_write_json(manifest_path, manifest)
        history.append({"from": "RUNNING", "to": "SUCCEEDED", "at_utc": _utcnow()})
        _status(exp_dir, "SUCCEEDED", history)
        atomic_write_json(
            exp_dir / "DONE.json",
            {"status": "SUCCEEDED", "exp_id": cfg.exp_id, "resumed": True},
        )
        _append_log(exp_dir, "RESUME_SUCCEEDED\n")
    except BaseException as exc:
        manifest["ended_at_utc"] = _utcnow()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        history.append({"from": "RUNNING", "to": "FAILED", "at_utc": _utcnow()})
        _status(exp_dir, "FAILED", history)
        atomic_write_json(
            exp_dir / "FAILED.json",
            {"status": "FAILED", "exp_id": cfg.exp_id, "error": manifest["error"]},
        )
        _append_log(exp_dir, f"RESUME_ERROR:\n{traceback.format_exc()}")
        traceback.print_exc()
        return 1

    print(
        "RESUME_SUCCEEDED "
        f"exp_id={cfg.exp_id} completed_units={resume_event['completed_units_before_resume']} "
        f"recommendation={json.dumps(metrics['recommended_winner_pending_reviewer'], ensure_ascii=False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
