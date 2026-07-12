"""Deterministic, offline evaluations for Chulk agents.

The runner always injects :class:`~chulk.testing.ScriptedLLMClient`. Scenarios
therefore exercise the real agent loop, tools, state, and trace callbacks
without reading provider configuration or making network calls.

Example::

    from chulk.evals import EvalExpectations, EvalScenario, run_eval

    scenario = EvalScenario(
        name="ready",
        user_message="Are you ready?",
        scripted_responses=({"type": "final_answer", "content": "Ready."},),
        expectations=EvalExpectations(
            answer="Ready.",
            status="completed",
            tool_sequence=(),
            trace_event_sequence=("model_request_started", "final_answer", "turn_finished"),
        ),
    )
    run_eval(scenario).assert_passed()

Trace expectations are matched as an ordered subsequence. This keeps an eval
focused on meaningful lifecycle events while allowing unrelated diagnostics to
be added between them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from chulk.core import Agent as CoreAgent
from chulk.testing import ScriptedLLMClient, ScriptedResponse


EvalAgentFactory: TypeAlias = Callable[[ScriptedLLMClient], object]


@dataclass(frozen=True)
class EvalExpectations:
    """Provider-neutral outcomes expected from one agent turn.

    A ``None`` field is not checked. Tool sequences are exact; trace-event
    sequences are ordered subsequences of all events emitted by the turn.
    """

    answer: str | None = None
    status: str | None = "completed"
    tool_sequence: tuple[str, ...] | None = None
    trace_event_sequence: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tool_sequence is not None:
            object.__setattr__(self, "tool_sequence", tuple(self.tool_sequence))
        if self.trace_event_sequence is not None:
            object.__setattr__(self, "trace_event_sequence", tuple(self.trace_event_sequence))


@dataclass(frozen=True)
class EvalScenario:
    """One deterministic user turn and its scripted model actions."""

    name: str
    user_message: str
    scripted_responses: tuple[ScriptedResponse, ...]
    expectations: EvalExpectations

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("EvalScenario.name cannot be empty")
        if not self.user_message.strip():
            raise ValueError("EvalScenario.user_message cannot be empty")
        object.__setattr__(self, "scripted_responses", tuple(self.scripted_responses))


@dataclass(frozen=True)
class EvalResult:
    """Inspectable actual outcomes and assertion failures for one scenario."""

    scenario_name: str
    passed: bool
    answer: str | None
    status: str | None
    tool_sequence: tuple[str, ...]
    trace_event_sequence: tuple[str, ...]
    responses_remaining: int
    failures: tuple[str, ...] = ()
    exception: str | None = None

    def assert_passed(self) -> None:
        """Raise one readable assertion containing every observed mismatch."""
        if self.passed:
            return
        details = "\n".join(f"- {failure}" for failure in self.failures)
        raise AssertionError(f"Eval scenario {self.scenario_name!r} failed:\n{details}")


class EvalRunner:
    """Run scenarios with a fresh scripted client and optional agent factory."""

    def __init__(self, agent_factory: EvalAgentFactory | None = None) -> None:
        self.agent_factory = agent_factory

    def run(self, scenario: EvalScenario, *, agent: object | None = None) -> EvalResult:
        """Run one scenario against an existing agent or a factory-created one.

        For an existing agent, its model client and trace callback are restored
        after the turn. An agent created by this runner is closed after results
        have been captured.
        """
        if agent is not None and self.agent_factory is not None:
            raise ValueError("Pass an agent or configure an agent_factory, not both")

        client = ScriptedLLMClient(scenario.scripted_responses)
        instance: object | None = agent
        runtime: Any | None = None
        owned_agent = agent is None
        original_client: object | None = None
        original_callback: object | None = None
        turn_count = 0
        answer: str | None = None
        execution_exception: Exception | None = None
        cleanup_exception: Exception | None = None
        events: list[str] = []

        try:
            if instance is None:
                factory = self.agent_factory or CoreAgent
                instance = factory(client)
            runtime = _runtime_from_agent(instance)
            turn_count = len(runtime.state.turns)
            original_client = runtime.llm_client
            original_callback = runtime.event_callback
            runtime.llm_client = client

            def capture_event(event_type: str, payload: dict[str, Any]) -> None:
                events.append(event_type)
                if callable(original_callback):
                    original_callback(event_type, payload)

            runtime.event_callback = capture_event
            answer = _run_agent(instance, scenario.user_message)
        except Exception as exc:  # The result records execution failures for batch evals.
            execution_exception = exc
        finally:
            if runtime is not None:
                runtime.event_callback = original_callback
                runtime.llm_client = original_client
            if owned_agent and instance is not None:
                close = getattr(instance, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # pragma: no cover - defensive cleanup accounting
                        cleanup_exception = exc

        turn = _newest_turn(runtime, turn_count)
        status = getattr(turn, "status", None)
        tool_sequence = _tool_sequence(turn)
        failures = _evaluate(
            scenario.expectations,
            answer=answer,
            status=status,
            tool_sequence=tool_sequence,
            trace_event_sequence=tuple(events),
        )
        if execution_exception is not None:
            failures.insert(0, f"execution raised {_format_exception(execution_exception)}")
        if cleanup_exception is not None:
            failures.append(f"agent cleanup raised {_format_exception(cleanup_exception)}")

        return EvalResult(
            scenario_name=scenario.name,
            passed=not failures,
            answer=answer,
            status=status,
            tool_sequence=tool_sequence,
            trace_event_sequence=tuple(events),
            responses_remaining=client.remaining,
            failures=tuple(failures),
            exception=_format_exception(execution_exception) if execution_exception else None,
        )

    def run_many(self, scenarios: Iterable[EvalScenario]) -> tuple[EvalResult, ...]:
        """Run each scenario with a fresh agent and return all results."""
        return tuple(self.run(scenario) for scenario in scenarios)


def run_eval(
    scenario: EvalScenario,
    *,
    agent: object | None = None,
    agent_factory: EvalAgentFactory | None = None,
) -> EvalResult:
    """Run one deterministic scenario with a concise functional API."""
    return EvalRunner(agent_factory).run(scenario, agent=agent)


def _runtime_from_agent(agent: object) -> Any:
    runtime = getattr(agent, "runtime", agent)
    required = ("state", "llm_client", "event_callback")
    missing = [attribute for attribute in required if not hasattr(runtime, attribute)]
    if missing:
        raise TypeError(f"Eval agent runtime is missing: {', '.join(missing)}")
    return runtime


def _run_agent(agent: object, user_message: str) -> str:
    run_turn = getattr(agent, "run_turn", None)
    if callable(run_turn):
        answer = run_turn(user_message)
    else:
        run = getattr(agent, "run", None)
        if not callable(run):
            raise TypeError("Eval agent must expose run_turn(message) or run(message)")
        answer = run(user_message)
    if not isinstance(answer, str):
        raise TypeError("Eval agent must return an answer string")
    return answer


def _newest_turn(runtime: Any | None, previous_count: int) -> Any | None:
    if runtime is None:
        return None
    turns = runtime.state.turns
    return turns[-1] if len(turns) > previous_count else None


def _tool_sequence(turn: Any | None) -> tuple[str, ...]:
    if turn is None:
        return ()
    return tuple(record.resolved_tool_name or record.tool_name for record in turn.tool_calls)


def _evaluate(
    expected: EvalExpectations,
    *,
    answer: str | None,
    status: str | None,
    tool_sequence: tuple[str, ...],
    trace_event_sequence: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    if expected.answer is not None and answer != expected.answer:
        failures.append(f"expected answer {expected.answer!r}, got {answer!r}")
    if expected.status is not None and status != expected.status:
        failures.append(f"expected status {expected.status!r}, got {status!r}")
    if expected.tool_sequence is not None and tool_sequence != expected.tool_sequence:
        failures.append(f"expected tool sequence {expected.tool_sequence!r}, got {tool_sequence!r}")
    if expected.trace_event_sequence is not None and not _is_ordered_subsequence(
        expected.trace_event_sequence,
        trace_event_sequence,
    ):
        failures.append(
            "expected trace-event subsequence "
            f"{expected.trace_event_sequence!r}, got {trace_event_sequence!r}"
        )
    return failures


def _is_ordered_subsequence(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    expected_index = 0
    for event_type in actual:
        if expected_index < len(expected) and event_type == expected[expected_index]:
            expected_index += 1
    return expected_index == len(expected)


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "EvalAgentFactory",
    "EvalExpectations",
    "EvalResult",
    "EvalRunner",
    "EvalScenario",
    "run_eval",
]
