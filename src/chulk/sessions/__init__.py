"""SQLite-backed conversation session persistence."""

from chulk.sessions.models import (
    ConversationRecord,
    ConversationSummaryRecord,
    MessageRecord,
    SessionHit,
    SessionMessage,
    SessionSearchPage,
    SessionWindow,
)
from chulk.sessions.recorder import SessionRecorder
from chulk.sessions.async_recorder import AsyncSessionRecorder
from chulk.sessions.external_recorder import (
    AsyncExternalTranscriptRecorder,
    ExternalTranscriptRecorder,
)
from chulk.sessions.search import (
    MAX_SESSION_QUERY_CHARS,
    MAX_SESSION_QUERY_TERMS,
    MAX_SESSION_SEARCH_LIMIT,
    MAX_SESSION_SNIPPET_CHARS,
    MAX_SESSION_WINDOW_LIMIT,
    MAX_SESSION_WINDOW_RADIUS,
    SessionSearchService,
)
from chulk.sessions.sqlite_store import AmbiguousSessionError, SessionNotFoundError, SQLiteSessionStore

__all__ = [
    "AmbiguousSessionError",
    "ConversationRecord",
    "ConversationSummaryRecord",
    "MAX_SESSION_QUERY_CHARS",
    "MAX_SESSION_QUERY_TERMS",
    "MAX_SESSION_SEARCH_LIMIT",
    "MAX_SESSION_SNIPPET_CHARS",
    "MAX_SESSION_WINDOW_LIMIT",
    "MAX_SESSION_WINDOW_RADIUS",
    "MessageRecord",
    "SessionHit",
    "SessionMessage",
    "SessionSearchPage",
    "SessionSearchService",
    "SessionWindow",
    "SQLiteSessionStore",
    "SessionNotFoundError",
    "SessionRecorder",
    "AsyncSessionRecorder",
    "AsyncExternalTranscriptRecorder",
    "ExternalTranscriptRecorder",
]
