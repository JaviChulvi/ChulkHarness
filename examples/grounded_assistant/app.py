"""Small source-grounded assistant embedded with the public Chulk SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Iterable

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import scripted_or_live  # noqa: E402
from chulk import Agent, Capabilities, RunResult, Tool, ToolContext, ToolEffect, ToolPolicy  # noqa: E402
from chulk.testing import ScriptedResponse  # noqa: E402


Source = dict[str, str]
CITATION = re.compile(r"\[source:([A-Za-z0-9._-]+)\]")
CITATION_LIKE = re.compile(r"\[source:[^\]]*(?:\]|$)")
READ_POLICY = ToolPolicy(version="1.0.0", effect=ToolEffect.READ)


@dataclass
class ScopedSources:
    """Sources the host already authorized for one user and workspace."""

    items: dict[str, Source]
    retrieved_ids: set[str] = field(default_factory=set)


# Tiny fictional fixture for this example; it is not customer data.
FICTIONAL_SOURCES = {
    "handbook-leave": {
        "title": "Fictional Northstar Labs leave handbook",
        "text": "Employees receive 20 days of annual leave per calendar year.",
    }
}


@Tool(policy=READ_POLICY)
def get_source(source_id: str, context: ToolContext[ScopedSources]) -> Source:
    """Retrieve one exact source from the host-authorized source set."""
    sources = context.require_deps()
    if source_id not in sources.items:
        raise ValueError("source_id is not available in the authenticated scope")
    sources.retrieved_ids.add(source_id)
    return {"source_id": source_id, **sources.items[source_id]}


def _validate_citations(answer: str, sources: ScopedSources) -> tuple[str, ...]:
    if any(CITATION.fullmatch(marker) is None for marker in CITATION_LIKE.findall(answer)):
        raise ValueError("answer contains an invalid citation marker; expected [source:ID]")
    citations = tuple(dict.fromkeys(CITATION.findall(answer)))
    if not citations:
        raise ValueError("answer must include at least one [source:ID] citation")
    rejected = sorted(set(citations) - sources.retrieved_ids)
    if rejected:
        raise ValueError(
            "citation IDs were not retrieved from the scoped source set: "
            + ", ".join(rejected)
        )
    return citations


def run_assistant(
    question: str,
    *,
    sources: ScopedSources | None = None,
    responses: Iterable[ScriptedResponse] | None = None,
) -> tuple[RunResult, ScopedSources, tuple[str, ...], str]:
    scoped = sources or ScopedSources(dict(FICTIONAL_SOURCES))
    scoped.retrieved_ids.clear()
    scripted = responses or [
        {
            "type": "tool_call",
            "tool_name": "get_source",
            "arguments": {"source_id": "handbook-leave"},
        },
        {
            "type": "final_answer",
            "content": "Annual leave is 20 days [source:handbook-leave].",
        },
    ]
    config, llm, mode = scripted_or_live("grounded_assistant", scripted)
    available = ", ".join(sorted(scoped.items))
    prompt = (
        f"Authorized source IDs: {available}. Use get_source before answering. "
        "Answer only from retrieved evidence and cite it as [source:ID]. "
        f"Question: {question}"
    )
    with Agent(
        config=config,
        llm=llm,
        tools=[get_source],
        skills=[],
        deps=scoped,
        capabilities=Capabilities.none(),
    ) as assistant:
        result = assistant.run_result(prompt)
    citations = _validate_citations(result.content, scoped)
    return result, scoped, citations, mode


def main() -> None:
    result, sources, citations, mode = run_assistant("How much annual leave is offered?")
    print(f"mode: {mode} (scripted is NOT live answer-quality proof)")
    print("answer: " + result.content)
    print("evidence:")
    for source_id in citations:
        source = sources.items[source_id]
        print(f"- [source:{source_id}] {source['title']}: {source['text']}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
