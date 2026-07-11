"""External-style static typing fixture for the installed SDK contract."""

from pathlib import Path

from chulk import Agent, AgentConfig, Skills, Tool, Tools


assert Tool is not None

config: AgentConfig = AgentConfig.local(
    project_root=Path.cwd(),
    runtime_dir=".chulk",
    permission_profile="read-only",
)

agent = Agent(
    config=config,
    tools=[Tools.calculator],
    skills=[Skills.files],
)

result: str = agent.run("Calculate 2 + 2")
