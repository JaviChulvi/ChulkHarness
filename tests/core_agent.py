"""Internal core-agent construction helper for focused runtime tests."""

from typing import Any

from chulk._runtime.request import AgentAssemblyRequest
from chulk.config import Config
from chulk.core import Agent
from chulk.core.action_runtime import AgentRuntimeComponents
from chulk.llm import LLMClient
from chulk.runtime import create_agent


def build_core_agent(llm_client: LLMClient, **kwargs: Any) -> Agent:
    return Agent(AgentRuntimeComponents(llm_client=llm_client, **kwargs))


def create_runtime_agent(
    config: Config,
    llm_client_factory: Any = None,
    **kwargs: Any,
) -> Agent:
    return create_agent(
        AgentAssemblyRequest(
            config=config,
            llm_client_factory=llm_client_factory,
            **kwargs,
        )
    )
