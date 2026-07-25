"""Bounded synchronous, parallel, and detached child-task execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread

from chulk.children.factory import ChildAgentFactory, ChildAgentRunner
from chulk.children.models import (
    ChildCompletionDelivery,
    ChildTask,
    ChildTaskClaim,
)
from chulk.children.service import (
    ChildResultRejectedError,
    DelegationPolicy,
    ParentCompletionValidator,
)
from chulk.children.store import (
    DEFAULT_CHILD_LEASE_SECONDS,
    ChildTaskLeaseConflictError,
    ChildTaskStore,
)
from chulk.usage import BudgetExceededError


CompletionConsumer = Callable[
    [ChildCompletionDelivery, ChildTask],
    None,
]


@dataclass(frozen=True, slots=True)
class ChildSupervisorHandle:
    """Handle for one in-process detached worker pool."""

    _stop: Event
    _threads: tuple[Thread, ...]

    def stop(self, *, timeout: float | None = None) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)


class TaskSupervisor:
    """Lease, fence, heartbeat, and execute children with bounded workers."""

    def __init__(
        self,
        store: ChildTaskStore,
        agent_factory: ChildAgentFactory,
        *,
        worker_id: str,
        policy: DelegationPolicy | None = None,
        validator: ParentCompletionValidator | None = None,
        lease_seconds: int = DEFAULT_CHILD_LEASE_SECONDS,
        heartbeat_seconds: float | None = None,
        event_callback: Callable[[str, ChildTask], None] | None = None,
    ) -> None:
        clean_worker = worker_id.strip()
        if not clean_worker:
            raise ValueError("child supervisor worker_id cannot be empty")
        if lease_seconds < 2:
            raise ValueError("child supervisor lease_seconds must be at least two")
        self.store = store
        self.agent_factory = agent_factory
        self.worker_id = clean_worker
        self.policy = policy or DelegationPolicy()
        self.validator = validator or ParentCompletionValidator()
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or max(0.5, lease_seconds / 3)
        self.event_callback = event_callback
        self._active: dict[str, ChildAgentRunner] = {}
        self._active_lock = Lock()

    def run_once(self) -> ChildTask | None:
        claim = self.store.claim_next(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if claim is None:
            return None
        return self._run_claim(claim)

    def run_task(self, task_id: str) -> ChildTask:
        task = self.store.get(task_id)
        claim = self.store.claim(
            task.id,
            expected_revision=task.revision,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        return self._run_claim(claim)

    def run_batch(self, *, max_tasks: int | None = None) -> tuple[ChildTask, ...]:
        limit = max_tasks or self.policy.max_parallel_workers
        if limit < 1:
            raise ValueError("max_tasks must be positive")
        worker_count = min(limit, self.policy.max_parallel_workers)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(self.run_once) for _ in range(limit)]
        return tuple(
            task
            for future in futures
            if (task := future.result()) is not None
        )

    def start_detached(
        self,
        *,
        workers: int | None = None,
        poll_seconds: float = 0.25,
    ) -> ChildSupervisorHandle:
        worker_count = workers or self.policy.max_parallel_workers
        if not 1 <= worker_count <= self.policy.max_parallel_workers:
            raise ValueError("detached worker count exceeds supervisor policy")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.store.recover_expired(actor=f"{self.worker_id}:startup")
        stop = Event()
        threads = tuple(
            Thread(
                target=self._worker_loop,
                args=(stop, poll_seconds),
                name=f"chulk-child-{self.worker_id}-{index}",
                daemon=True,
            )
            for index in range(worker_count)
        )
        for thread in threads:
            thread.start()
        return ChildSupervisorHandle(stop, threads)

    def cancel_active(self, task: ChildTask) -> None:
        """Cancel a live runtime after durable cancellation intent is recorded."""
        with self._active_lock:
            runner = self._active.get(task.id)
        if runner is not None:
            runner.cancel()

    def deliver_once(
        self,
        consumer: CompletionConsumer,
        *,
        lease_seconds: int | None = None,
    ) -> ChildCompletionDelivery | None:
        delivery = self.store.claim_delivery(
            worker_id=f"{self.worker_id}:delivery",
            lease_seconds=lease_seconds or self.lease_seconds,
        )
        if delivery is None:
            return None
        try:
            consumer(delivery, self.store.get(delivery.task_id))
        except Exception as exc:
            return self.store.finish_delivery(
                delivery,
                delivered=False,
                error=str(exc),
            )
        return self.store.finish_delivery(delivery, delivered=True)

    def _worker_loop(self, stop: Event, poll_seconds: float) -> None:
        while not stop.is_set():
            try:
                task = self.run_once()
            except Exception:
                task = None
            if task is None:
                stop.wait(poll_seconds)

    def _run_claim(self, claim: ChildTaskClaim) -> ChildTask:
        task = self.store.get(claim.task_id)
        runner: ChildAgentRunner | None = None
        heartbeat: _AttemptHeartbeat | None = None
        try:
            runner = self.agent_factory.create(task, claim)
            with self._active_lock:
                self._active[task.id] = runner
            heartbeat = _AttemptHeartbeat(
                self.store,
                claim,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_seconds,
                on_lost=runner.cancel,
            )
            heartbeat.start()
            result = runner.run(
                boundary=lambda: self.store.assert_attempt_boundary(claim)
            )
            heartbeat.stop()
            if heartbeat.error is not None:
                raise heartbeat.error
            self.validator.require_valid(task, result)
            completed = self.store.complete(claim, result)
            self._emit("child.completed", completed)
            return completed
        except BudgetExceededError as exc:
            failed = self.store.exhaust_budget(claim, str(exc))
            self._emit("child.budget_exhausted", failed)
            return failed
        except ChildResultRejectedError as exc:
            failed = self.store.fail(
                claim,
                "Parent validation rejected child completion: " + str(exc),
            )
            self._emit("child.failed", failed)
            return failed
        except ChildTaskLeaseConflictError:
            current = self.store.get(claim.task_id)
            if current.cancellation_requested:
                cancelled = self.store.fail(
                    claim,
                    "Child execution stopped at a cancellation boundary.",
                )
                self._emit("child.cancelled", cancelled)
                return cancelled
            raise
        except Exception as exc:
            failed = self.store.fail(claim, str(exc))
            self._emit("child.failed", failed)
            return failed
        finally:
            if heartbeat is not None:
                heartbeat.stop()
            with self._active_lock:
                self._active.pop(task.id, None)
            if runner is not None:
                runner.close()

    def _emit(self, kind: str, task: ChildTask) -> None:
        if self.event_callback is not None:
            self.event_callback(kind, task)


class _AttemptHeartbeat:
    def __init__(
        self,
        store: ChildTaskStore,
        claim: ChildTaskClaim,
        *,
        lease_seconds: int,
        interval_seconds: float,
        on_lost: Callable[[], None],
    ) -> None:
        self.store = store
        self.claim = claim
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self.on_lost = on_lost
        self.error: ChildTaskLeaseConflictError | None = None
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name=f"chulk-child-heartbeat-{claim.task_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not current_thread():
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.claim = self.store.heartbeat(
                    self.claim,
                    lease_seconds=self.lease_seconds,
                )
            except ChildTaskLeaseConflictError as exc:
                self.error = exc
                self.on_lost()
                return


__all__ = [
    "ChildSupervisorHandle",
    "CompletionConsumer",
    "TaskSupervisor",
]
