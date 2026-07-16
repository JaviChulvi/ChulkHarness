"""Narrow service port consumed by action-loop transport drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from chulk.core.model_transport import ModelTransport
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects


class ActionLoopPort(Protocol):
    """The complete, explicit dependency surface of the action loop."""

    model: ModelTransport
    tools: ToolExecutor
    effects: TurnEffects


@dataclass
class ActionLoopRuntime:
    """Concrete runtime assembled by Agent from focused services."""

    model: ModelTransport
    tools: ToolExecutor
    effects: TurnEffects
