"""Public errors raised by durable run persistence."""


class RunNotFoundError(LookupError):
    """Raised when a run is absent or outside the caller's authority."""


class RunConflictError(RuntimeError):
    """Raised when an idempotency or optimistic transition conflicts."""


class RunLeaseError(RuntimeError):
    """Raised when a worker does not own a live run lease."""


class InvalidRunTransitionError(ValueError):
    """Raised when a requested run transition is not valid."""


class EffectConflictError(RuntimeError):
    """Raised when an external effect cannot be repeated safely."""


__all__ = [
    "EffectConflictError",
    "InvalidRunTransitionError",
    "RunConflictError",
    "RunLeaseError",
    "RunNotFoundError",
]
