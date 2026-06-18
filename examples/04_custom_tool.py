"""Define typed custom tools with the public @Tool decorator."""

from __future__ import annotations

from typing import Annotated, Literal

import bootstrap  # noqa: F401
from chulk import Agent, Tool

from common import live_config, print_run_result


CUSTOMERS = {
    "cus_001": {"name": "Ada Lovelace", "tier": "enterprise", "active_seats": 42},
    "cus_002": {"name": "Grace Hopper", "tier": "startup", "active_seats": 8},
}


@Tool
def lookup_customer(
    customer_id: Annotated[str, "Customer id such as cus_001 or cus_002"],
) -> dict:
    """Look up a customer account by id."""
    return CUSTOMERS.get(customer_id, {"error": f"Unknown customer id: {customer_id}"})


@Tool
def recommend_discount(
    tier: Literal["enterprise", "startup", "standard"],
    active_seats: Annotated[int, "Current number of active seats"],
) -> dict:
    """Recommend a renewal discount from customer tier and seat count."""
    if tier == "enterprise" and active_seats >= 25:
        return {"discount_percent": 12, "reason": "large enterprise renewal"}
    if tier == "startup":
        return {"discount_percent": 8, "reason": "startup growth incentive"}
    return {"discount_percent": 3, "reason": "standard retention offer"}


def main() -> None:
    assistant = Agent(
        config=live_config("04-custom-tool"),
        tools=[lookup_customer, recommend_discount],
        skills=[],
    )
    result = assistant.run_result(
        "Look up customer cus_001, recommend a renewal discount, and explain the reasoning."
    )
    print_run_result(result)


if __name__ == "__main__":
    main()
