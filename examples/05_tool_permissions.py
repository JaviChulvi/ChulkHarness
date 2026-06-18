"""Approve or deny confirming tools with a permission callback."""

import bootstrap  # noqa: F401
from chulk import Agent, PermissionDecisionRecord, PermissionRequest, Tool, ToolPermissionLevel

from common import live_config, print_run_result


@Tool(
    permission_level=ToolPermissionLevel.EXTERNAL_SERVICE,
    requires_confirmation=True,
)
def enrich_company(company: str) -> dict:
    """Pretend to call an external enrichment service for a company."""
    return {
        "company": company,
        "industry": "developer tools",
        "risk": "low",
        "source": "example fixture, not a live service",
    }


def approve_external_service(request: PermissionRequest, record: PermissionDecisionRecord) -> bool:
    print("approval requested")
    print(f"tool: {request.tool_name}")
    print(f"permission: {request.permission_level.value}")
    print(f"policy decision: {record.decision.value}")
    return request.tool_name == "enrich_company"


def main() -> None:
    assistant = Agent(
        config=live_config("05-tool-permissions", permission_profile="workspace-write"),
        tools=[enrich_company],
        skills=[],
        permission_callback=approve_external_service,
    )
    result = assistant.run_result("Use the enrichment tool for ChulkHarness and summarize the result.")
    print_run_result(result)


if __name__ == "__main__":
    main()
