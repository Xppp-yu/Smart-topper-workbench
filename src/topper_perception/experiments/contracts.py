"""Experiment config contracts: schema validation, EXP-ID legality, state machine.

This module is the single source of truth for what a runnable experiment config
looks like and which state transitions are legal. It mirrors the declarative
``experiment_v0.1.schema.json`` with a stdlib-only runtime check so the runner
does not need a ``jsonschema`` dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

#: Schema identifier written to ``schema_version`` in experiment configs.
SCHEMA_VERSION = "experiment-v0.1"

#: Allowed experiment scopes. ``smoke`` is the cheap deterministic run;
#: ``mini``/``full`` are the heavier runs that require a clean worktree.
SCOPES = ("smoke", "mini", "full")

#: Implemented runner types. New runners register here; unknown types fail.
RUNNER_TYPES = ("dummy",)

#: Fields every experiment config must provide.
REQUIRED_FIELDS = (
    "schema_version",
    "exp_id",
    "task_id",
    "scope",
    "runner_type",
    "seed",
    "output_root",
    "parameters",
)

#: EXP-ID legality: ``EXP-`` followed by letters/digits/dot/underscore/dash.
#: Forbids whitespace and path separators so an EXP-ID is always a safe dir name.
EXP_ID_RE = re.compile(r"^EXP-[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExperimentError(Exception):
    """Base class for all experiment-governance errors."""


class ConfigValidationError(ExperimentError):
    """The experiment config failed schema/contract validation."""


class ExpIdError(ExperimentError):
    """The EXP-ID is illegal or already exists."""


class StateTransitionError(ExperimentError):
    """An illegal state-machine transition was attempted."""


class DirtyWorktreeError(ExperimentError):
    """A mini/full run was requested with a dirty Git worktree."""


@dataclass(frozen=True)
class ExperimentConfig:
    """A validated, frozen experiment config."""

    schema_version: str
    exp_id: str
    task_id: str
    scope: str
    runner_type: str
    seed: int
    output_root: str
    parameters: Mapping[str, Any]
    raw: Mapping[str, Any]


def validate_exp_id(exp_id: str) -> None:
    """Raise :class:`ExpIdError` when ``exp_id`` is not a legal EXP-ID."""
    if not isinstance(exp_id, str) or not exp_id:
        raise ExpIdError("exp_id must be a non-empty string.")
    if EXP_ID_RE.fullmatch(exp_id) is None:
        raise ExpIdError(
            f"exp_id {exp_id!r} is illegal; expected the form "
            "EXP-<ALNUM>... using only letters, digits, '.', '_' and '-'."
        )


def validate_experiment_config(config: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a raw config mapping and return a frozen :class:`ExperimentConfig`."""
    if not isinstance(config, Mapping):
        raise ConfigValidationError("Experiment config must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ConfigValidationError(f"Config missing required fields: {missing}")

    schema_version = str(config["schema_version"])
    if schema_version != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}."
        )

    exp_id = str(config["exp_id"])
    validate_exp_id(exp_id)

    task_id = str(config["task_id"]).strip()
    if not task_id:
        raise ConfigValidationError("task_id must be a non-empty string.")

    scope = str(config["scope"]).lower()
    if scope not in SCOPES:
        raise ConfigValidationError(f"scope must be one of {SCOPES}; got {scope!r}.")

    runner_type = str(config["runner_type"]).lower()
    if runner_type not in RUNNER_TYPES:
        raise ConfigValidationError(
            f"Unknown runner_type {runner_type!r}; known: {sorted(RUNNER_TYPES)}."
        )

    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigValidationError("seed must be an integer.")

    output_root = str(config["output_root"]).strip()
    if not output_root:
        raise ConfigValidationError("output_root must be a non-empty string.")

    parameters = config["parameters"]
    if not isinstance(parameters, Mapping):
        raise ConfigValidationError("parameters must be a JSON object.")

    return ExperimentConfig(
        schema_version=schema_version,
        exp_id=exp_id,
        task_id=task_id,
        scope=scope,
        runner_type=runner_type,
        seed=seed,
        output_root=output_root,
        parameters=dict(parameters),
        raw=dict(config),
    )


class State(StrEnum):
    """Experiment lifecycle states (governance plan section 5)."""

    DRAFT = "DRAFT"
    CODE_READY = "CODE_READY"
    SMOKE_PASS = "SMOKE_PASS"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


#: Legal single-step transitions. Terminal states (SUCCEEDED/FAILED) have no exit.
ALLOWED_TRANSITIONS: dict[State, frozenset[State]] = {
    State.DRAFT: frozenset({State.CODE_READY}),
    State.CODE_READY: frozenset({State.SMOKE_PASS}),
    State.SMOKE_PASS: frozenset({State.QUEUED}),
    State.QUEUED: frozenset({State.RUNNING}),
    State.RUNNING: frozenset({State.SUCCEEDED, State.FAILED}),
    State.SUCCEEDED: frozenset(),
    State.FAILED: frozenset(),
}


def transition(current: State, nxt: State) -> State:
    """Return ``nxt`` if ``current -> nxt`` is legal; else raise.

    This encodes both required chains from the governance plan:
    ``DRAFT -> CODE_READY -> SMOKE_PASS`` and
    ``QUEUED -> RUNNING -> SUCCEEDED | FAILED``.
    """
    allowed = ALLOWED_TRANSITIONS.get(current)
    if allowed is None:
        raise StateTransitionError(f"Unknown state {current!r}.")
    if nxt not in allowed:
        raise StateTransitionError(
            f"Illegal transition {current!r} -> {nxt!r}; allowed: {sorted(allowed)}."
        )
    return nxt
