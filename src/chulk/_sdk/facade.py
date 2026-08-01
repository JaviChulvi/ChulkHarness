"""Compatibility exports and factories for the public SDK facade."""

from __future__ import annotations

from typing import Any

from chulk._sdk.agent import Agent
from chulk._sdk.async_agent import AsyncAgent
from chulk._sdk.config import ensure_chat_kwargs
from chulk._sdk.handles import AgentHandle, AsyncAgentHandle
from chulk._sdk.hosted import (
    AsyncHostedRuntime as AsyncHostedRuntime,
    HostedRuntime as HostedRuntime,
)


def agent(**kwargs: Any) -> Agent:
    """Create the public synchronous Agent facade."""
    return Agent(**kwargs)


def chat_agent(**kwargs: Any) -> Agent:
    """Create a plain chat Agent with no tools or skills configured."""
    ensure_chat_kwargs(kwargs)
    return Agent(tools=[], skills=[], **kwargs)


def ChatAgent(**kwargs: Any) -> Agent:
    """Compatibility constructor for a plain chat Agent."""
    return chat_agent(**kwargs)


def async_agent(**kwargs: Any) -> AsyncAgent:
    """Create the public asynchronous Agent facade."""
    return AsyncAgent(**kwargs)


def async_chat_agent(**kwargs: Any) -> AsyncAgent:
    """Create an async plain chat Agent with no tools or skills configured."""
    ensure_chat_kwargs(kwargs)
    return AsyncAgent(tools=[], skills=[], **kwargs)


def AsyncChatAgent(**kwargs: Any) -> AsyncAgent:
    """Compatibility constructor for an async plain chat Agent."""
    return async_chat_agent(**kwargs)


__all__ = [
    "Agent",
    "AgentHandle",
    "AsyncAgent",
    "AsyncAgentHandle",
    "AsyncChatAgent",
    "AsyncHostedRuntime",
    "ChatAgent",
    "HostedRuntime",
    "agent",
    "async_agent",
    "async_chat_agent",
    "chat_agent",
]
