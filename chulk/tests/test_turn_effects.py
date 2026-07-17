"""Contracts for consuming reducer outcomes at the mutation boundary."""

from __future__ import annotations

import pytest

from chulk.core.transitions import TransitionOutcome
from chulk.core.turn_effects import PendingReflection, _validate_application


@pytest.mark.parametrize(
    ("outcome", "response", "pending"),
    [
        (TransitionOutcome.STOP, None, None),
        (
            TransitionOutcome.AWAIT_RESULT,
            None,
            None,
        ),
        (TransitionOutcome.CONTINUE, "unexpected", None),
        (
            TransitionOutcome.PROCEED,
            None,
            PendingReflection(proposed_answer="unexpected"),
        ),
    ],
)
def test_transition_application_rejects_mismatched_reducer_outcome(
    outcome: TransitionOutcome,
    response: str | None,
    pending: PendingReflection | None,
) -> None:
    with pytest.raises(RuntimeError):
        _validate_application(outcome=outcome, response=response, pending=pending)


def test_transition_application_preserves_reducer_outcome() -> None:
    application = _validate_application(
        outcome=TransitionOutcome.STOP,
        response="done",
        pending=None,
    )

    assert application.outcome is TransitionOutcome.STOP
    assert application.response == "done"
