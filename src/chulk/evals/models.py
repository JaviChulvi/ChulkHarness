"""Immutable public contracts for Chulk agent evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from chulk.events import AgentEvent
from chulk.hosting import ExecutionScope
from chulk.results import RunResult, plain_data
from chulk.testing import ScriptedResponse


EVAL_DATASET_SCHEMA_VERSION = 1


class EvaluationMode(StrEnum):
    SCRIPTED = "scripted"
    REPLAY = "replay"
    LIVE = "live"


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _required_text(value: str, name: str) -> str:
    selected = value.strip()
    if not selected:
        raise ValueError(f"{name} cannot be empty")
    return selected


@dataclass(frozen=True)
class EvalReference:
    """Typed ground truth consumed by built-in and custom graders."""

    answer: str | None = None
    contains: tuple[str, ...] = ()
    regex: str | None = None
    json_schema: Mapping[str, Any] | None = None
    status: str | None = "completed"
    tool_sequence: tuple[str, ...] | None = None
    tool_arguments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    tool_results: Mapping[str, Any] = field(default_factory=dict)
    tool_failures: tuple[str, ...] | None = None
    event_sequence: tuple[str, ...] | None = None
    skill_names: tuple[str, ...] | None = None
    memory_ids: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contains", tuple(self.contains))
        object.__setattr__(self, "tool_arguments", _mapping(self.tool_arguments))
        object.__setattr__(self, "tool_results", _mapping(self.tool_results))
        object.__setattr__(self, "metadata", _mapping(self.metadata))
        if self.json_schema is not None:
            object.__setattr__(self, "json_schema", _mapping(self.json_schema))
        for name in ("tool_sequence", "tool_failures", "event_sequence", "skill_names", "memory_ids"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalReference":
        known = {
            "answer", "contains", "regex", "json_schema", "status", "tool_sequence",
            "tool_arguments", "tool_results", "tool_failures", "event_sequence",
            "skill_names", "memory_ids", "metadata",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown eval reference fields: {', '.join(unknown)}")
        return cls(
            answer=_optional_string(value.get("answer"), "reference.answer"),
            contains=_string_tuple(value.get("contains", ()), "reference.contains"),
            regex=_optional_string(value.get("regex"), "reference.regex"),
            json_schema=_optional_mapping(value.get("json_schema"), "reference.json_schema"),
            status=_optional_string(value.get("status", "completed"), "reference.status"),
            tool_sequence=_optional_string_tuple(value.get("tool_sequence"), "reference.tool_sequence"),
            tool_arguments=_nested_mapping(value.get("tool_arguments", {}), "reference.tool_arguments"),
            tool_results=_required_mapping(value.get("tool_results", {}), "reference.tool_results"),
            tool_failures=_optional_string_tuple(value.get("tool_failures"), "reference.tool_failures"),
            event_sequence=_optional_string_tuple(value.get("event_sequence"), "reference.event_sequence"),
            skill_names=_optional_string_tuple(value.get("skill_names"), "reference.skill_names"),
            memory_ids=_optional_string_tuple(value.get("memory_ids"), "reference.memory_ids"),
            metadata=_required_mapping(value.get("metadata", {}), "reference.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class EvalTurn:
    """One user input and optional scripted model actions."""

    input: str
    scripted_responses: tuple[ScriptedResponse, ...] = ()
    reference: EvalReference | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _required_text(self.input, "EvalTurn.input"))
        object.__setattr__(self, "scripted_responses", tuple(self.scripted_responses))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalTurn":
        known = {"input", "scripted_responses", "reference", "metadata"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown eval turn fields: {', '.join(unknown)}")
        raw_input = value.get("input")
        if not isinstance(raw_input, str):
            raise ValueError("eval turn input must be a string")
        raw_responses = value.get("scripted_responses", ())
        if not isinstance(raw_responses, list | tuple):
            raise ValueError("eval turn scripted_responses must be an array")
        raw_reference = value.get("reference")
        reference = None
        if raw_reference is not None:
            reference = EvalReference.from_dict(
                _required_mapping(raw_reference, "turn.reference")
            )
        return cls(
            input=raw_input,
            scripted_responses=tuple(raw_responses),
            reference=reference,
            metadata=_required_mapping(value.get("metadata", {}), "turn.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class EvalCase:
    """One isolated, potentially multi-turn evaluation case."""

    id: str
    turns: tuple[EvalTurn, ...]
    reference: EvalReference | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fixture: str | None = None
    replay_fixture: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "EvalCase.id"))
        object.__setattr__(self, "turns", tuple(self.turns))
        if not self.turns:
            raise ValueError("EvalCase.turns cannot be empty")
        normalized_tags = tuple(dict.fromkeys(_required_text(tag, "EvalCase.tags item") for tag in self.tags))
        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(self, "metadata", _mapping(self.metadata))
        if self.fixture is not None:
            object.__setattr__(self, "fixture", _required_text(self.fixture, "EvalCase.fixture"))
        if self.replay_fixture is not None:
            object.__setattr__(
                self, "replay_fixture", _safe_relative_path(self.replay_fixture, "EvalCase.replay_fixture")
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalCase":
        known = {
            "schema_version", "id", "turns", "reference", "tags", "metadata",
            "fixture", "replay_fixture",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown eval case fields: {', '.join(unknown)}")
        version = value.get("schema_version", EVAL_DATASET_SCHEMA_VERSION)
        if version != EVAL_DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported eval dataset schema_version: {version}")
        raw_id = value.get("id")
        if not isinstance(raw_id, str):
            raise ValueError("eval case id must be a string")
        raw_turns = value.get("turns")
        if not isinstance(raw_turns, list | tuple):
            raise ValueError("eval case turns must be an array")
        raw_reference = value.get("reference")
        return cls(
            id=raw_id,
            turns=tuple(
                EvalTurn.from_dict(_required_mapping(turn, "case.turns item"))
                for turn in raw_turns
            ),
            reference=(
                EvalReference.from_dict(_required_mapping(raw_reference, "case.reference"))
                if raw_reference is not None else None
            ),
            tags=_string_tuple(value.get("tags", ()), "case.tags"),
            metadata=_required_mapping(value.get("metadata", {}), "case.metadata"),
            fixture=_optional_string(value.get("fixture"), "case.fixture"),
            replay_fixture=_optional_string(value.get("replay_fixture"), "case.replay_fixture"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": EVAL_DATASET_SCHEMA_VERSION, **plain_data(self)}


@dataclass(frozen=True)
class EvalDataset:
    """A validated collection of uniquely identified cases."""

    cases: tuple[EvalCase, ...]
    source: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases:
            raise ValueError("EvalDataset.cases cannot be empty")
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate eval case ids: {', '.join(duplicates)}")
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source).resolve())

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "EvalDataset":
        source = Path(path).expanduser().resolve()
        cases: list[EvalCase] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            try:
                cases.append(EvalCase.from_dict(_required_mapping(payload, "eval case")))
            except ValueError as exc:
                raise ValueError(f"invalid eval case at {source}:{line_number}: {exc}") from exc
        return cls(tuple(cases), source=source)

    @property
    def digest(self) -> str:
        payload = [case.to_dict() for case in self.cases]
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def filtered(self, *, tags: tuple[str, ...] = ()) -> "EvalDataset":
        selected = frozenset(tags)
        if not selected:
            return self
        return EvalDataset(tuple(case for case in self.cases if selected <= set(case.tags)), self.source)


@dataclass(frozen=True)
class MetricThreshold:
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("MetricThreshold requires min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("MetricThreshold min cannot exceed max")

    def accepts(self, value: float | int | None) -> bool:
        if value is None:
            return False
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


@dataclass(frozen=True)
class EvalSafetyPolicy:
    allowed_tool_names: tuple[str, ...] = ()
    allow_unknown_cost: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tool_names", tuple(self.allowed_tool_names))


@dataclass(frozen=True)
class EvalContext:
    suite_name: str
    target_name: str
    case_id: str
    trial: int
    workspace: Path
    mode: EvaluationMode
    scope: ExecutionScope
    llm: object | None = None
    deps: object | None = None
    provider: str | None = None
    model: str | None = None
    safety: EvalSafetyPolicy = field(default_factory=EvalSafetyPolicy)


AgentFactory: TypeAlias = Callable[[EvalContext], object]
FixtureFactory: TypeAlias = Callable[[EvalContext], object]


@dataclass(frozen=True)
class EvalTarget:
    name: str
    agent_factory: AgentFactory
    provider: str | None = None
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "EvalTarget.name"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True)
class EvalSuite:
    name: str
    dataset: EvalDataset
    targets: tuple[EvalTarget, ...]
    graders: tuple[object, ...] = ()
    mode: EvaluationMode = EvaluationMode.SCRIPTED
    trials: int = 1
    concurrency: int = 1
    timeout_seconds: float = 120.0
    required_graders: tuple[str, ...] = ()
    thresholds: Mapping[str, MetricThreshold] = field(default_factory=dict)
    fixtures: Mapping[str, FixtureFactory] = field(default_factory=dict)
    safety: EvalSafetyPolicy = field(default_factory=EvalSafetyPolicy)
    max_total_cost: float | None = None
    fail_fast: bool = False
    store: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "EvalSuite.name"))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "graders", tuple(self.graders))
        object.__setattr__(self, "required_graders", tuple(self.required_graders))
        object.__setattr__(self, "thresholds", _mapping(self.thresholds))
        object.__setattr__(self, "fixtures", _mapping(self.fixtures))
        object.__setattr__(self, "mode", EvaluationMode(self.mode))
        if not self.targets:
            raise ValueError("EvalSuite.targets cannot be empty")
        if any(not callable(target.agent_factory) for target in self.targets):
            raise TypeError("EvalTarget.agent_factory must be callable")
        target_names = [target.name for target in self.targets]
        if len(target_names) != len(set(target_names)):
            raise ValueError("EvalSuite target names must be unique")
        grader_names = [str(getattr(grader, "name", "")) for grader in self.graders]
        if any(not name for name in grader_names) or len(grader_names) != len(set(grader_names)):
            raise ValueError("EvalSuite grader names must be non-empty and unique")
        unknown_required = sorted(set(self.required_graders) - set(grader_names))
        if unknown_required:
            raise ValueError("unknown required graders: " + ", ".join(unknown_required))
        unknown_fixtures = sorted(
            {case.fixture for case in self.dataset.cases if case.fixture is not None}
            - set(self.fixtures)
        )
        if unknown_fixtures:
            raise ValueError("unknown eval fixtures: " + ", ".join(unknown_fixtures))
        if self.mode is EvaluationMode.REPLAY:
            missing_replays = [case.id for case in self.dataset.cases if case.replay_fixture is None]
            if missing_replays:
                raise ValueError("replay cases require replay_fixture: " + ", ".join(missing_replays))
        if self.trials < 1:
            raise ValueError("EvalSuite.trials must be greater than zero")
        if self.concurrency < 1:
            raise ValueError("EvalSuite.concurrency must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("EvalSuite.timeout_seconds must be greater than zero")
        if any(not isinstance(threshold, MetricThreshold) for threshold in self.thresholds.values()):
            raise TypeError("EvalSuite thresholds must be MetricThreshold values")
        if self.max_total_cost is not None and self.max_total_cost < 0:
            raise ValueError("EvalSuite.max_total_cost cannot be negative")
        if self.mode is EvaluationMode.LIVE and self.max_total_cost is None:
            raise ValueError("live eval suites require max_total_cost")


@dataclass(frozen=True)
class GradeResult:
    grader: str
    score: float
    passed: bool
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grader", _required_text(self.grader, "GradeResult.grader"))
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("GradeResult.score must be between 0 and 1")
        object.__setattr__(self, "details", _mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class EvalTurnResult:
    index: int
    result: RunResult
    events: tuple[AgentEvent, ...]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "result": self.result.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class TrialResult:
    case_id: str
    target_name: str
    trial: int
    turns: tuple[EvalTurnResult, ...]
    duration_seconds: float
    grades: tuple[GradeResult, ...] = ()
    exception: str | None = None
    workspace: Path | None = None

    @property
    def final_result(self) -> RunResult | None:
        return self.turns[-1].result if self.turns else None

    @property
    def passed(self) -> bool:
        required = [grade for grade in self.grades if grade.details.get("required")]
        return self.exception is None and all(grade.passed for grade in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target_name": self.target_name,
            "trial": self.trial,
            "turns": [turn.to_dict() for turn in self.turns],
            "duration_seconds": self.duration_seconds,
            "grades": [grade.to_dict() for grade in self.grades],
            "exception": self.exception,
            "workspace": str(self.workspace) if self.workspace is not None else None,
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    target_name: str
    trials: tuple[TrialResult, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target_name": self.target_name,
            "trials": [trial.to_dict() for trial in self.trials],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EvalReport:
    id: str
    suite_name: str
    dataset_digest: str
    started_at: str
    ended_at: str
    cases: tuple[CaseResult, ...]
    metrics: Mapping[str, float]
    threshold_failures: tuple[str, ...] = ()
    operational_errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _mapping(self.metrics))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def passed(self) -> bool:
        return not self.threshold_failures and not self.operational_errors

    def assert_thresholds(self) -> None:
        if self.operational_errors:
            raise RuntimeError("Evaluation failed operationally:\n" + "\n".join(self.operational_errors))
        if self.threshold_failures:
            raise AssertionError("Evaluation quality gate failed:\n" + "\n".join(self.threshold_failures))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "suite_name": self.suite_name,
            "dataset_digest": self.dataset_digest,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
            "metrics": dict(self.metrics),
            "threshold_failures": list(self.threshold_failures),
            "operational_errors": list(self.operational_errors),
            "metadata": plain_data(self.metadata),
        }


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _required_mapping(value, name)


def _nested_mapping(value: Any, name: str) -> Mapping[str, Mapping[str, Any]]:
    raw = _required_mapping(value, name)
    return {str(key): _required_mapping(item, f"{name}.{key}") for key, item in raw.items()}


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be an array")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return tuple(value)


def _optional_string_tuple(value: Any, name: str) -> tuple[str, ...] | None:
    return None if value is None else _string_tuple(value, name)


def _safe_relative_path(value: str, name: str) -> str:
    path = Path(_required_text(value, name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must remain inside the dataset directory")
    return path.as_posix()


__all__ = [
    "AgentFactory", "CaseResult", "EVAL_DATASET_SCHEMA_VERSION",
    "EvalCase", "EvalContext", "EvalDataset", "EvalReference", "EvalReport",
    "EvalSafetyPolicy", "EvalSuite", "EvalTarget", "EvalTurn", "EvalTurnResult",
    "EvaluationMode", "FixtureFactory", "GradeResult", "MetricThreshold", "TrialResult",
]
