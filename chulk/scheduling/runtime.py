"""Fresh-profile execution and supervision for durable automation runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Protocol

from chulk.core import Agent
from chulk.gateway import DeliveryTarget
from chulk.profiles import ProfileRuntimeFactory
from chulk.scheduling.models import (
    AutomationDeliveryState,
    AutomationRun,
    ScheduledJob,
)
from chulk.scheduling.store import (
    DEFAULT_AUTOMATION_LEASE_SECONDS,
    SQLiteScheduleStore,
)
from chulk.scheduling.triggers import AutomationCompletionBridge
from chulk.usage import BudgetExceededError, UsageDimensions


@dataclass(frozen=True, slots=True)
class AutomationExecutionResult:
    summary: str
    trace_id: str | None = None
    usage: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    artifact_refs: tuple[str, ...] = ()


class AutomationRunner(Protocol):
    """Host-owned execution boundary for one claimed job."""

    def run(
        self,
        job: ScheduledJob,
        run: AutomationRun,
    ) -> AutomationExecutionResult:
        """Resolve current authority and execute one occurrence."""

    def cancel(self) -> None:
        """Cancel owned runtime resources."""

    def close(self) -> None:
        """Release owned runtime resources."""


class RuntimeAutomationRunner:
    """Build a fresh agent from the current profile for every occurrence."""

    def __init__(
        self,
        runtime_factory: ProfileRuntimeFactory,
        *,
        llm_client_factory: Callable | None = None,
        agent_kwargs_factory: Callable[[ScheduledJob], dict[str, Any]] | None = None,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.llm_client_factory = llm_client_factory
        self.agent_kwargs_factory = agent_kwargs_factory or (lambda _job: {})
        self._agent: Agent | None = None

    def run(
        self,
        job: ScheduledJob,
        run: AutomationRun,
    ) -> AutomationExecutionResult:
        if run.job_id != job.id or run.profile_id != job.profile_id:
            raise ValueError("automation run does not match its job")
        kwargs = dict(self.agent_kwargs_factory(job))
        forbidden = {
            "profile_id",
            "run_budget",
            "usage_dimensions",
            "conversation_metadata",
        }.intersection(kwargs)
        if forbidden:
            raise ValueError(
                "automation runner overrides host-owned fields: "
                + ", ".join(sorted(forbidden))
            )
        kwargs.update(
            {
                "conversation_id": None,
                "conversation_metadata": {
                    "profile_id": job.profile_id,
                    "automation_job_id": job.id,
                    "automation_run_id": run.id,
                    "automation_reason": run.reason.value,
                },
                "run_budget": job.budget,
                "usage_dimensions": UsageDimensions(
                    profile_id=job.profile_id,
                    channel=job.adapter,
                    job_id=job.id,
                ),
                "runtime_metadata": {
                    "automation_job_id": job.id,
                    "automation_run_id": run.id,
                    "automation_occurrence_at": run.occurrence_at.isoformat(),
                },
            }
        )
        if self.llm_client_factory is not None:
            kwargs["llm_client_factory"] = self.llm_client_factory
        self._agent = self.runtime_factory.create_agent(job.profile_id, **kwargs)
        response = self._agent.run_turn(
            job.prompt,
            extension_metadata={
                "source": "automation",
                "automation_job_id": job.id,
                "automation_run_id": run.id,
            },
        )
        turn = self._agent.state.turns[-1] if self._agent.state.turns else None
        usage = dict(turn.model_usage_totals) if turn is not None else {}
        return AutomationExecutionResult(
            summary=response,
            trace_id=self._agent.state.conversation_id,
            usage=usage,
            cost=(dict(usage["cost"]) if isinstance(usage.get("cost"), dict) else {}),
            artifact_refs=(
                (str(self._agent.trace_logger.path),)
                if self._agent.trace_logger is not None
                else ()
            ),
        )

    def cancel(self) -> None:
        if self._agent is not None:
            self._agent.close()

    def close(self) -> None:
        if self._agent is not None:
            self._agent.close()
            self._agent = None


AutomationRunnerFactory = Callable[[ScheduledJob, AutomationRun], AutomationRunner]
AutomationDelivery = Callable[
    [DeliveryTarget, AutomationExecutionResult, ScheduledJob, AutomationRun],
    None,
]


@dataclass(frozen=True, slots=True)
class AutomationSupervisorHandle:
    _stop: Event
    _thread: Thread

    def stop(self, *, timeout: float | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


class AutomationSupervisor:
    """Lease, heartbeat, execute, record, and optionally deliver due work."""

    def __init__(
        self,
        store: SQLiteScheduleStore,
        runner_factory: AutomationRunnerFactory,
        *,
        worker_id: str,
        delivery: AutomationDelivery | None = None,
        lease_seconds: int = DEFAULT_AUTOMATION_LEASE_SECONDS,
        heartbeat_seconds: float | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("automation worker_id cannot be empty")
        if lease_seconds < 2:
            raise ValueError("automation lease_seconds must be at least two")
        self.store = store
        self.runner_factory = runner_factory
        self.worker_id = worker_id.strip()
        self.delivery = delivery
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or max(0.5, lease_seconds / 3)
        self._active: AutomationRunner | None = None

    def run_once(self, *, adapter: str | None = None) -> AutomationRun | None:
        jobs = self.store.claim_due(
            adapter=adapter,
            limit=1,
            lease_seconds=self.lease_seconds,
            worker_id=self.worker_id,
        )
        if not jobs:
            return None
        job = jobs[0]
        assert job.active_run_id is not None
        run = self.store.get_run(job.active_run_id)
        runner = self.runner_factory(job, run)
        self._active = runner
        heartbeat = _AutomationHeartbeat(
            self.store,
            job.id,
            run.claim_token or "",
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_seconds,
            on_lost=runner.cancel,
        )
        heartbeat.start()
        try:
            result = runner.run(job, run)
            heartbeat.stop()
            if heartbeat.error is not None:
                raise heartbeat.error
            delivery_state = AutomationDeliveryState.NONE
            if self.delivery is not None:
                delivery_state = AutomationDeliveryState.PENDING
            completed = self.store.complete(
                job.id,
                run.claim_token or "",
                result={"summary": result.summary},
                trace_id=result.trace_id,
                usage=result.usage or {},
                cost=result.cost or {},
                artifact_refs=result.artifact_refs,
                delivery_state=delivery_state,
            )
            if not completed:
                return self.store.get_run(run.id)
            if self.delivery is not None:
                try:
                    self.delivery(job.target, result, job, run)
                except Exception as exc:
                    self.store.mark_delivery(
                        run.id,
                        AutomationDeliveryState.RETRYABLE,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    self.store.mark_delivery(
                        run.id,
                        AutomationDeliveryState.DELIVERED,
                    )
            completed_run = self.store.get_run(run.id)
            AutomationCompletionBridge(self.store).job_completed(completed_run)
            return completed_run
        except BudgetExceededError as exc:
            heartbeat.stop()
            self.store.fail(
                job.id,
                run.claim_token or "",
                str(exc),
                budget_exhausted=True,
            )
            return self.store.get_run(run.id)
        except Exception as exc:
            heartbeat.stop()
            self.store.fail(
                job.id,
                run.claim_token or "",
                f"{type(exc).__name__}: {exc}",
            )
            return self.store.get_run(run.id)
        finally:
            heartbeat.stop()
            runner.close()
            self._active = None

    def run_batch(
        self,
        *,
        limit: int = 10,
        adapter: str | None = None,
    ) -> tuple[AutomationRun, ...]:
        if limit < 1:
            raise ValueError("automation batch limit must be positive")
        completed: list[AutomationRun] = []
        for _ in range(limit):
            run = self.run_once(adapter=adapter)
            if run is None:
                break
            completed.append(run)
        return tuple(completed)

    def start_detached(
        self,
        *,
        adapter: str | None = None,
        poll_seconds: float = 0.5,
    ) -> AutomationSupervisorHandle:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.store.recover_expired(actor=f"{self.worker_id}:startup")
        stop = Event()
        thread = Thread(
            target=self._worker_loop,
            args=(stop, adapter, poll_seconds),
            name=f"chulk-automation-{self.worker_id}",
            daemon=True,
        )
        thread.start()
        return AutomationSupervisorHandle(stop, thread)

    def cancel_active(self) -> None:
        if self._active is not None:
            self._active.cancel()

    def _worker_loop(
        self,
        stop: Event,
        adapter: str | None,
        poll_seconds: float,
    ) -> None:
        while not stop.is_set():
            try:
                self.store.recover_expired(actor=f"{self.worker_id}:recovery")
                run = self.run_once(adapter=adapter)
            except Exception:
                run = None
            if run is None:
                stop.wait(poll_seconds)


class _AutomationHeartbeat:
    def __init__(
        self,
        store: SQLiteScheduleStore,
        job_id: str,
        claim_token: str,
        *,
        lease_seconds: int,
        interval_seconds: float,
        on_lost: Callable[[], None],
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.claim_token = claim_token
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self.on_lost = on_lost
        self.error: RuntimeError | None = None
        self._stop = Event()
        self._thread = Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if self.store.renew_lease(
                self.job_id,
                self.claim_token,
                lease_seconds=self.lease_seconds,
            ):
                continue
            self.error = RuntimeError("automation execution lease was lost")
            try:
                self.on_lost()
            except Exception:
                pass
            return


def runtime_automation_runner_factory(
    runtime_factory: ProfileRuntimeFactory,
    *,
    llm_client_factory: Callable | None = None,
    agent_kwargs_factory: Callable[[ScheduledJob], dict[str, Any]] | None = None,
) -> AutomationRunnerFactory:
    def create(_job: ScheduledJob, _run: AutomationRun) -> AutomationRunner:
        return RuntimeAutomationRunner(
            runtime_factory,
            llm_client_factory=llm_client_factory,
            agent_kwargs_factory=agent_kwargs_factory,
        )

    return create


__all__ = [
    "AutomationDelivery",
    "AutomationExecutionResult",
    "AutomationRunner",
    "AutomationRunnerFactory",
    "AutomationSupervisor",
    "AutomationSupervisorHandle",
    "RuntimeAutomationRunner",
    "runtime_automation_runner_factory",
]
