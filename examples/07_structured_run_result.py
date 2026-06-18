"""Use run_result() when an application needs structured metadata."""

import bootstrap  # noqa: F401
from chulk import Agent, Tool

from common import live_config, print_run_result


@Tool
def ticket_priority(severity: str, affected_customers: int) -> dict:
    """Classify a support ticket priority."""
    if severity.lower() == "critical" or affected_customers >= 10:
        return {"priority": "p0", "sla_hours": 1}
    if severity.lower() == "high":
        return {"priority": "p1", "sla_hours": 4}
    return {"priority": "p2", "sla_hours": 24}


def main() -> None:
    assistant = Agent(
        config=live_config("07-structured-run-result"),
        tools=[ticket_priority],
        skills=[],
    )
    result = assistant.run_result(
        "Classify a high severity ticket affecting 3 customers, then explain what the app should display."
    )
    print_run_result(result)
    print("\n=== JSON-friendly Result ===")
    print(result.to_dict())


if __name__ == "__main__":
    main()
