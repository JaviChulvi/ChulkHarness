"""Factories that assemble fresh, narrowed child-agent runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Protocol

from chulk.children.models import ChildTask, ChildTaskClaim, ChildTaskResult
from chulk.core import Agent
from chulk.core.context import TurnContextSection
from chulk.execution import ExecutionBackend
from chulk.mcp import MCPServerConfig
from chulk.profiles import ProfileRuntimeFactory
from chulk.usage import RunBudget, UsageDimensions


AttemptBoundary = Callable[[], ChildTask]
ChildResultBuilder = Callable[[ChildTask, Agent, str], ChildTaskResult]


class ChildAgentRunner(Protocol):
    """One fresh child context owned by a single durable attempt."""

    def run(self, *, boundary: AttemptBoundary) -> ChildTaskResult:
        """Execute the explicit child package and return evidence."""

    def cancel(self) -> None:
        """Stop owned resources after durable cancellation."""

    def close(self) -> None:
        """Finalize the child runtime."""


class ChildAgentFactory(Protocol):
    """Host boundary for building a child with narrower runtime authority."""

    def create(
        self,
        task: ChildTask,
        claim: ChildTaskClaim,
    ) -> ChildAgentRunner:
        """Create a fresh context for exactly one claimed attempt."""


class RuntimeChildAgentFactory:
    """Create real Chulk agents from a host-owned profile runtime factory."""

    def __init__(
        self,
        profile_factory: ProfileRuntimeFactory,
        *,
        tool_catalog: Mapping[str, object] | None = None,
        mcp_catalog: Mapping[str, MCPServerConfig] | None = None,
        backend_factories: Mapping[
            str,
            Callable[[ChildTask], ExecutionBackend],
        ]
        | None = None,
        llm_client_factory: Callable | None = None,
        result_builder: ChildResultBuilder | None = None,
        additional_budget_resolver: Callable[
            [ChildTask],
            tuple[RunBudget, ...],
        ]
        | None = None,
    ) -> None:
        self.profile_factory = profile_factory
        self.tool_catalog = dict(tool_catalog or {})
        self.mcp_catalog = dict(mcp_catalog or {})
        self.backend_factories = dict(backend_factories or {})
        self.llm_client_factory = llm_client_factory
        self.result_builder = result_builder or _default_result
        self.additional_budget_resolver = (
            additional_budget_resolver or (lambda _task: ())
        )

    def create(
        self,
        task: ChildTask,
        claim: ChildTaskClaim,
    ) -> ChildAgentRunner:
        if claim.task_id != task.id or claim.profile_id != task.profile_id:
            raise ValueError("child claim does not match the requested task")
        resolved = self.profile_factory.resolve(task.profile_id)
        profile = resolved.profile
        if (
            task.spec.model_profile_id is not None
            and task.spec.model_profile_id != profile.model_profile_id
        ):
            raise ValueError("child model profile does not match its owner profile")
        missing_tools = sorted(set(task.spec.tool_names) - self.tool_catalog.keys())
        if missing_tools:
            raise ValueError(
                "child tool catalog is missing: " + ", ".join(missing_tools)
            )
        missing_mcp = sorted(
            set(task.spec.mcp_server_labels) - self.mcp_catalog.keys()
        )
        if missing_mcp:
            raise ValueError(
                "child MCP catalog is missing: " + ", ".join(missing_mcp)
            )
        backend = self._backend(task, profile.execution_backend_id)
        kwargs: dict[str, object] = {
            "conversation_id": None,
            "conversation_metadata": {
                "profile_id": task.profile_id,
                "child_task_id": task.id,
                "child_attempt_id": claim.attempt_id,
                "parent_conversation_id": task.parent_conversation_id,
                "parent_turn_id": task.parent_turn_id,
                "parent_trace_id": task.parent_trace_id,
            },
            "tool_specs": tuple(
                self.tool_catalog[name] for name in task.spec.tool_names
            ),
            "mcp_servers": tuple(
                self.mcp_catalog[label]
                for label in task.spec.mcp_server_labels
            ),
            "capabilities": task.spec.capabilities,
            "allowed_skill_names": task.spec.skill_names,
            "run_budget": task.spec.budget,
            "additional_run_budgets": self.additional_budget_resolver(task),
            "usage_dimensions": UsageDimensions(
                profile_id=task.profile_id,
                goal_id=task.goal_id,
                child_task_id=task.id,
            ),
            "runtime_metadata": {
                "child_task_id": task.id,
                "child_attempt_id": claim.attempt_id,
                "parent_task_id": task.lineage.parent_task_id,
                "root_task_id": task.lineage.root_task_id or task.id,
            },
        }
        if backend is not None:
            kwargs["execution_backend"] = backend
        if self.llm_client_factory is not None:
            kwargs["llm_client_factory"] = self.llm_client_factory
        agent = self.profile_factory.create_agent(task.profile_id, **kwargs)
        return _RuntimeChildAgentRunner(
            task=task,
            agent=agent,
            result_builder=self.result_builder,
        )

    def _backend(
        self,
        task: ChildTask,
        profile_backend_name: str,
    ) -> ExecutionBackend | None:
        factory = self.backend_factories.get(task.spec.backend_name)
        if factory is not None:
            return factory(task)
        if task.spec.backend_name != profile_backend_name:
            raise ValueError(
                f"child execution backend {task.spec.backend_name!r} "
                "is not configured by the host"
            )
        return None


class _RuntimeChildAgentRunner:
    def __init__(
        self,
        *,
        task: ChildTask,
        agent: Agent,
        result_builder: ChildResultBuilder,
    ) -> None:
        self.task = task
        self.agent = agent
        self.result_builder = result_builder

    def run(self, *, boundary: AttemptBoundary) -> ChildTaskResult:
        boundary()
        context = self.task.spec.to_dict()["context"]
        sections = (
            [
                TurnContextSection(
                    id=f"child:{self.task.id}:delegated-context",
                    title="Explicit delegated context",
                    source="parent",
                    content=json.dumps(
                        context,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    metadata={
                        "child_task_id": self.task.id,
                        "parent_task_id": self.task.lineage.parent_task_id,
                    },
                )
            ]
            if context
            else None
        )
        response = self.agent.run_turn(
            self.task.spec.instruction,
            context_sections=sections,
            extension_metadata={
                "child_task_id": self.task.id,
                "child_attempt": True,
            },
        )
        boundary()
        return self.result_builder(self.task, self.agent, response)

    def cancel(self) -> None:
        self.agent.close()

    def close(self) -> None:
        self.agent.close()


def _default_result(
    task: ChildTask,
    agent: Agent,
    response: str,
) -> ChildTaskResult:
    turn = agent.state.turns[-1] if agent.state.turns else None
    trace_path = (
        str(agent.trace_logger.path)
        if agent.trace_logger is not None
        else None
    )
    return ChildTaskResult(
        summary=response,
        structured_output={"response": response},
        artifact_refs=(trace_path,) if trace_path is not None else (),
        usage=turn.model_usage_totals if turn is not None else {},
        trace_id=agent.state.conversation_id,
    )


__all__ = [
    "AttemptBoundary",
    "ChildAgentFactory",
    "ChildAgentRunner",
    "ChildResultBuilder",
    "RuntimeChildAgentFactory",
]
