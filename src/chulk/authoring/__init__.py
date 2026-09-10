"""Portable agent authoring, publication, compilation, and runtime assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chulk._lazy import public_dir, resolve_export

if TYPE_CHECKING:
    from chulk.authoring.catalog import (
        AgentDefinitionRecord,
        AgentDefinitionStore,
        ArtifactCatalog,
        ArtifactAvailability,
        AsyncAgentDefinitionStore,
        AsyncInMemoryAgentDefinitionStore,
        EvaluationCaseResult,
        EvaluationReport,
        InMemoryAgentDefinitionStore,
        PromptCatalog,
        PublicationError,
        PublishedArtifact,
        PublishedPrompt,
        PublishedTool,
        ToolCatalog,
        ValidationFinding,
        ValidationReport,
        validate_definition_catalogs,
    )
    from chulk.authoring.compiler import (
        AgentCompiler,
        CompiledAgentPackage,
        CompilerRequest,
        ReviewPreview,
        WorkflowGenerator,
    )
    from chulk.authoring.directory import AgentDirectory, initialize_agent_directory
    from chulk.authoring.models import (
        AGENT_DEFINITION_SCHEMA_VERSION,
        AgentDefinition,
        BudgetDefinition,
        DefinitionProvenance,
        DefinitionStatus,
        ToolReference,
        TriggerDefinition,
        VersionedReference,
        WorkflowApproval,
        WorkflowEffect,
        WorkflowGraph,
        WorkflowStep,
        canonical_json,
        request_digest,
    )
    from chulk.authoring.runtime import (
        AgentDefinitionRuntime,
        PublishedSkillResolver,
        ResolvedAgentDefinition,
    )


__all__ = [
    "AGENT_DEFINITION_SCHEMA_VERSION",
    "AgentCompiler",
    "AgentDirectory",
    "AgentDefinition",
    "AgentDefinitionRecord",
    "AgentDefinitionRuntime",
    "AgentDefinitionStore",
    "ArtifactAvailability",
    "ArtifactCatalog",
    "AsyncAgentDefinitionStore",
    "AsyncInMemoryAgentDefinitionStore",
    "BudgetDefinition",
    "CompiledAgentPackage",
    "CompilerRequest",
    "DefinitionProvenance",
    "DefinitionStatus",
    "EvaluationCaseResult",
    "EvaluationReport",
    "InMemoryAgentDefinitionStore",
    "PromptCatalog",
    "PublicationError",
    "PublishedArtifact",
    "PublishedPrompt",
    "PublishedSkillResolver",
    "PublishedTool",
    "ResolvedAgentDefinition",
    "ReviewPreview",
    "ToolCatalog",
    "ToolReference",
    "TriggerDefinition",
    "ValidationFinding",
    "ValidationReport",
    "VersionedReference",
    "WorkflowApproval",
    "WorkflowEffect",
    "WorkflowGenerator",
    "WorkflowGraph",
    "WorkflowStep",
    "canonical_json",
    "initialize_agent_directory",
    "request_digest",
    "validate_definition_catalogs",
]


_EXPORT_MODULES = (
    "chulk.authoring.models",
    "chulk.authoring.catalog",
    "chulk.authoring.compiler",
    "chulk.authoring.directory",
    "chulk.authoring.runtime",
)


if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        return resolve_export(
            name,
            public_names=__all__,
            owner_modules=_EXPORT_MODULES,
            namespace=globals(),
        )

    def __dir__() -> list[str]:
        return public_dir(__all__, globals())
