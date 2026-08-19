from __future__ import annotations

import pytest

from topper_perception.experiments.contracts import (
    ConfigValidationError,
    ExpIdError,
    State,
    StateTransitionError,
    transition,
    validate_exp_id,
    validate_experiment_config,
)


def _valid_config(**overrides):
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": "EXP-RUNNER-DUMMY-SMOKE-20260819-R01",
        "task_id": "TASK-EXP-RUNNER-B-v0.1",
        "scope": "smoke",
        "runner_type": "dummy",
        "seed": 42,
        "output_root": "outputs/experiments",
        "parameters": {"n_samples": 256},
    }
    config.update(overrides)
    return config


def test_valid_config_parses() -> None:
    cfg = validate_experiment_config(_valid_config())
    assert cfg.exp_id == "EXP-RUNNER-DUMMY-SMOKE-20260819-R01"
    assert cfg.scope == "smoke"
    assert cfg.seed == 42
    assert cfg.runner_type == "dummy"


@pytest.mark.parametrize(
    "field",
    ["schema_version", "exp_id", "task_id", "scope", "runner_type", "seed", "output_root", "parameters"],
)
def test_missing_required_field_raises(field: str) -> None:
    config = _valid_config()
    del config[field]
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_unsupported_schema_version_raises() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(schema_version="experiment-v99"))


def test_unknown_scope_raises() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(scope="huge"))


def test_unknown_runner_type_raises() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(runner_type="cnn"))


@pytest.mark.parametrize("seed", ["42", 42.0, True, None])
def test_seed_must_be_int(seed) -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(seed=seed))


def test_parameters_must_be_object() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(parameters=[1, 2, 3]))


def test_unknown_field_raises() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(surprise="boom"))


def test_data_manifests_optional_absent() -> None:
    cfg = validate_experiment_config(_valid_config())
    assert cfg.data_manifests == ()


def test_data_manifests_valid_normalizes_sha() -> None:
    cfg = validate_experiment_config(
        _valid_config(
            data_manifests=[
                {"path": "outputs/metrics/q.csv", "sha256": "9D" + "A" * 62},
            ]
        )
    )
    assert cfg.data_manifests == (
        {"path": "outputs/metrics/q.csv", "sha256": ("9d" + "a" * 62)},
    )


@pytest.mark.parametrize(
    "value",
    [
        "not-a-list",
        [{"path": "x"}],  # missing sha256
        [{"sha256": "a" * 64}],  # missing path
        [{"path": "", "sha256": "a" * 64}],  # empty path
        [{"path": "x", "sha256": "short"}],  # bad sha length
        [{"path": "x", "sha256": "z" * 64}],  # non-hex sha
        [{"path": 123, "sha256": "a" * 64}],  # non-string path
    ],
)
def test_data_manifests_invalid_raises(value) -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(data_manifests=value))


@pytest.mark.parametrize(
    "field",
    ["schema_version", "exp_id", "task_id", "scope", "runner_type", "output_root"],
)
@pytest.mark.parametrize("value", [42, None, True, ["x"]])
def test_string_fields_reject_non_string(field: str, value) -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(**{field: value}))


@pytest.mark.parametrize(
    "field",
    ["schema_version", "exp_id", "task_id", "scope", "runner_type", "output_root"],
)
def test_string_fields_reject_empty(field: str) -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(**{field: ""}))


def test_scope_is_case_sensitive() -> None:
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(_valid_config(scope="SMOKE"))


@pytest.mark.parametrize(
    "exp_id",
    [
        "EXP-P5.2-TINYCNN-SMOKE-20260819-R01",
        "EXP-RUNNER-DUMMY-SMOKE-20260819-R01",
        "EXP-A",
        "EXP-A1.b2_c3-d4",
    ],
)
def test_legal_exp_ids(exp_id: str) -> None:
    validate_exp_id(exp_id)  # must not raise


@pytest.mark.parametrize(
    "exp_id",
    [
        "",
        "EXP",
        "EXP-",
        "exp-lower",
        "EXP-..",
        "EXP-A/B",
        "EXP-A B",
        "EXP-A\\B",
        "smoke-1",
    ],
)
def test_illegal_exp_ids(exp_id: str) -> None:
    with pytest.raises(ExpIdError):
        validate_exp_id(exp_id)


def test_valid_state_transition_chain() -> None:
    state = transition(State.DRAFT, State.CODE_READY)
    state = transition(state, State.SMOKE_PASS)
    state = transition(state, State.QUEUED)
    state = transition(state, State.RUNNING)
    assert transition(state, State.SUCCEEDED) == State.SUCCEEDED


def test_valid_failed_transition() -> None:
    state = transition(State.QUEUED, State.RUNNING)
    assert transition(state, State.FAILED) == State.FAILED


@pytest.mark.parametrize(
    "current,nxt",
    [
        (State.DRAFT, State.QUEUED),
        (State.QUEUED, State.SUCCEEDED),
        (State.RUNNING, State.DRAFT),
        (State.SUCCEEDED, State.FAILED),
        (State.DRAFT, State.RUNNING),
    ],
)
def test_illegal_state_transitions_raise(current: State, nxt: State) -> None:
    with pytest.raises(StateTransitionError):
        transition(current, nxt)
