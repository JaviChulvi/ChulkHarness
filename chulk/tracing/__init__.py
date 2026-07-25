"""Tracing and logging primitives."""

from chulk.tracing.artifacts import (
    ArtifactAccessError,
    ArtifactRead,
    ArtifactRecord,
    TraceArtifactStore,
)
from chulk.tracing.logger import JSONLTraceLogger, TRACE_SCHEMA_VERSION, TraceEvent
from chulk.tracing.fixtures import (
    DEFAULT_REPLAY_FIXTURE_MAX_BYTES,
    ExpectedReplayOutcome,
    REPLAY_FIXTURE_SCHEMA_VERSION,
    RecordedModelAction,
    RecordedToolResult,
    ReplayFixture,
    ReplayFixtureError,
    ReplayFixtureSource,
    SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS,
    export_replay_fixture,
    load_replay_fixture,
    normalize_replay_value,
)
from chulk.tracing.reader import (
    DEFAULT_TRACE_MAX_BYTES,
    DEFAULT_TRACE_MAX_EVENTS,
    DEFAULT_TRACE_MAX_LINE_BYTES,
    LEGACY_TRACE_SCHEMA_VERSION,
    SUPPORTED_TRACE_SCHEMA_VERSIONS,
    Trace,
    TraceFormatError,
    TraceRecord,
)

__all__ = [
    "JSONLTraceLogger",
    "ArtifactAccessError",
    "ArtifactRead",
    "ArtifactRecord",
    "DEFAULT_REPLAY_FIXTURE_MAX_BYTES",
    "DEFAULT_TRACE_MAX_BYTES",
    "DEFAULT_TRACE_MAX_EVENTS",
    "DEFAULT_TRACE_MAX_LINE_BYTES",
    "ExpectedReplayOutcome",
    "LEGACY_TRACE_SCHEMA_VERSION",
    "REPLAY_FIXTURE_SCHEMA_VERSION",
    "RecordedModelAction",
    "RecordedToolResult",
    "ReplayFixture",
    "ReplayFixtureError",
    "ReplayFixtureSource",
    "SUPPORTED_TRACE_SCHEMA_VERSIONS",
    "SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS",
    "TRACE_SCHEMA_VERSION",
    "Trace",
    "TraceEvent",
    "TraceFormatError",
    "TraceRecord",
    "TraceArtifactStore",
    "export_replay_fixture",
    "load_replay_fixture",
    "normalize_replay_value",
]
