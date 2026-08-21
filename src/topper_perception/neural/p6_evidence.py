"""Frozen P6 / P6.1 evidence loader for the PoPu P7 runner.

Every P7 threshold must come from a machine-readable evidence file whose
SHA-256 is pinned in ``configs/analysis/popu_p7_robustness_v0.1.json``. The
runner rejects any drift between the on-disk file and its pinned SHA, and
refuses to use a hardcoded threshold in place of a verifiable source.

Two rule families are exposed:

- :class:`P6SingleRule` — the development-and-evaluation selected threshold
  produced by ``EXP-P6-POPU-REJECT`` for ``small_resnet`` at record level.
- :class:`P61EnsembleRule` — the calibrated three-repeat
  ``calibrated_mean_plus_unanimous`` rule produced by ``EXP-P6.1-POPU-CALIBRATION``
  for the same model. Per Reviewer point #2 (this round) the ensemble rule
  uses the unanimity branch (rules[1], ``threshold=0.5``,
  ``require_unanimous=true``), not the rules[0] pre-unanimity threshold.

Each rule carries its source path, pinned SHA-256, and the actual on-disk
SHA-256 so the review audit trail can confirm what was consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from topper_perception.experiments.artifacts import sha256_hex


@dataclass(frozen=True, slots=True)
class P6EvidenceFile:
    """A pinned P6 / P6.1 evidence file with its SHA-256 cross-check."""

    path: Path
    expected_sha256: str
    actual_sha256: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.expected_sha256.lower() != self.actual_sha256.lower():
            raise ValueError(
                f"P6 evidence SHA mismatch at {self.path}: "
                f"pinned={self.expected_sha256}, on-disk={self.actual_sha256}."
            )


def _resolve_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    """Resolve a JSON-pointer-style path such as ``/rules/1/threshold``."""
    if not pointer.startswith("/"):
        raise ValueError(f"P6 pointer must start with '/': {pointer!r}.")
    current: Any = payload
    for token in pointer[1:].split("/"):
        if token == "":
            continue
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(
                    f"P6 pointer {pointer!r} missing token {token!r}; "
                    f"available={sorted(current.keys())}."
                )
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"P6 pointer {pointer!r} expected integer index, got {token!r}."
                ) from exc
            if not 0 <= index < len(current):
                raise ValueError(
                    f"P6 pointer {pointer!r} index {index} out of range 0..{len(current) - 1}."
                )
            current = current[index]
        else:
            raise ValueError(
                f"P6 pointer {pointer!r} cannot descend into scalar at token {token!r}."
            )
    return current


def _coerce_threshold(value: Any, pointer: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"P6 threshold at {pointer!r} must not be a boolean.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"P6 threshold at {pointer!r} is not numeric: {value!r}."
        ) from exc
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"P6 threshold at {pointer!r} must lie in (0, 1], got {threshold}."
        )
    return threshold


def load_p6_evidence_file(
    path: Path | str,
    *,
    expected_sha256: str,
) -> P6EvidenceFile:
    """Read + SHA-verify one pinned P6 / P6.1 evidence file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"P6 evidence file not found: {resolved}.")
    raw = resolved.read_bytes()
    actual = sha256_hex(resolved)
    payload = _json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"P6 evidence file {resolved} must contain a JSON object at the root."
        )
    return P6EvidenceFile(
        path=resolved,
        expected_sha256=str(expected_sha256),
        actual_sha256=actual,
        payload=dict(payload),
    )


@dataclass(frozen=True, slots=True)
class P6SingleRule:
    """Frozen P6 single-checkpoint reject rule loaded from evidence."""

    threshold: float
    source: P6EvidenceFile
    threshold_pointer: str
    fallback_threshold_pointer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_kind": "p6_single",
            "threshold": float(self.threshold),
            "source_path": str(self.source.path),
            "source_expected_sha256": self.source.expected_sha256,
            "source_actual_sha256": self.source.actual_sha256,
            "threshold_pointer": self.threshold_pointer,
            "fallback_threshold_pointer": self.fallback_threshold_pointer,
        }


def load_p6_single_rule(config_block: Mapping[str, Any]) -> P6SingleRule:
    """Build the frozen P6 single-checkpoint rule from the P7 config block."""
    path = config_block["path"]
    expected_sha256 = str(config_block["expected_sha256"])
    threshold_pointer = str(config_block["threshold_pointer"])
    fallback_pointer = str(config_block["fallback_threshold_pointer"])
    source = load_p6_evidence_file(path, expected_sha256=expected_sha256)
    try:
        threshold = _coerce_threshold(
            _resolve_pointer(source.payload, threshold_pointer), threshold_pointer
        )
    except ValueError:
        threshold = _coerce_threshold(
            _resolve_pointer(source.payload, fallback_pointer), fallback_pointer
        )
        threshold_pointer = fallback_pointer
    return P6SingleRule(
        threshold=threshold,
        source=source,
        threshold_pointer=threshold_pointer,
        fallback_threshold_pointer=fallback_pointer,
    )


@dataclass(frozen=True, slots=True)
class P61EnsembleRule:
    """Frozen P6.1 ``calibrated_mean_plus_unanimous`` reject rule.

    Per Reviewer point #2 the canonical P6.1 rule is the unanimity branch
    (``rules[1]`` in the P6.1 evidence JSON): temperature = 0.75,
    threshold = 0.5, require_unanimous = true. The uncalibrated single-
    checkpoint threshold (rules[0] = 0.75) is never applied here.
    """

    temperature: float
    threshold: float
    require_unanimous: bool
    source: P6EvidenceFile
    temperature_pointer: str
    rule_pointer: str
    unanimity_field_pointer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_kind": "p6_1_ensemble",
            "temperature": float(self.temperature),
            "threshold": float(self.threshold),
            "require_unanimous": bool(self.require_unanimous),
            "source_path": str(self.source.path),
            "source_expected_sha256": self.source.expected_sha256,
            "source_actual_sha256": self.source.actual_sha256,
            "temperature_pointer": self.temperature_pointer,
            "rule_pointer": self.rule_pointer,
            "unanimity_field_pointer": self.unanimity_field_pointer,
        }


def load_p6_1_ensemble_rule(config_block: Mapping[str, Any]) -> P61EnsembleRule:
    """Build the frozen P6.1 ``calibrated_mean_plus_unanimous`` rule.

    The ``rule_pointer`` MUST point to ``/rules/1/threshold`` (the unanimity
    branch). Loading from ``/rules/0`` is rejected with a clear message
    because that threshold belongs to the pre-unanimity path and must not
    leak into the ensemble evaluation.
    """
    path = config_block["path"]
    expected_sha256 = str(config_block["expected_sha256"])
    temperature_pointer = str(config_block["temperature_pointer"])
    rule_pointer = str(config_block["rule_pointer"])
    unanimity_field_pointer = str(config_block["unanimous_require_field"])
    if not rule_pointer.endswith("/rules/1/threshold"):
        raise ValueError(
            f"P6.1 ensemble rule_pointer must end with '/rules/1/threshold' "
            f"(the unanimity branch); got {rule_pointer!r}. "
            "Per Reviewer point #2, rules[0] (pre-unanimity) must not be used."
        )
    source = load_p6_evidence_file(path, expected_sha256=expected_sha256)
    temperature = _coerce_threshold(
        _resolve_pointer(source.payload, temperature_pointer), temperature_pointer
    )
    threshold = _coerce_threshold(
        _resolve_pointer(source.payload, rule_pointer), rule_pointer
    )
    unanimity_required = _resolve_pointer(source.payload, unanimity_field_pointer)
    if not isinstance(unanimity_required, bool):
        raise ValueError(
            f"P6.1 unanimity flag at {unanimity_field_pointer!r} must be boolean; "
            f"got {unanimity_required!r}."
        )
    return P61EnsembleRule(
        temperature=temperature,
        threshold=threshold,
        require_unanimous=bool(unanimity_required),
        source=source,
        temperature_pointer=temperature_pointer,
        rule_pointer=rule_pointer,
        unanimity_field_pointer=unanimity_field_pointer,
    )


# Re-export json so the loader does not pick up the project's ``json`` shadow.
import json as _json  # noqa: E402