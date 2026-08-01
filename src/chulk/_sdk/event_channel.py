"""Per-run event delivery and serialized facade work gates."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import queue
import threading

from chulk.events import AgentEvent


_END = object()


class RunGate:
    """A ticketed lock that admits synchronous work in submission order."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving = 0

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._serving += 1
                self._condition.notify_all()


class RunEventChannel:
    """Bounded ordered channel owned by one generator run."""

    def __init__(self, *, max_events: int = 512) -> None:
        self._queue: queue.Queue[AgentEvent | object] = queue.Queue(maxsize=max_events)
        self._cancelled = threading.Event()
        self._finished = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def publish(self, event: AgentEvent) -> None:
        if self.cancelled or self._finished.is_set():
            return
        self._put(event)

    def finish(self, terminal: AgentEvent) -> None:
        if self._finished.is_set():
            return
        if not self.cancelled:
            self._put(terminal)
        self._finished.set()
        self._put(_END)

    def cancel(self) -> None:
        self._cancelled.set()
        self._put(_END, force=True)

    def get(self) -> AgentEvent | object:
        return self._queue.get()

    def iterate(self, worker: threading.Thread) -> Iterator[AgentEvent]:
        try:
            while True:
                item = self.get()
                if item is _END:
                    break
                if isinstance(item, AgentEvent):
                    yield item
        finally:
            self.cancel()
            worker.join()

    def _put(self, item: AgentEvent | object, *, force: bool = False) -> None:
        while force or not self.cancelled:
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                if force:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    continue
        return


__all__ = ["RunEventChannel", "RunGate"]
