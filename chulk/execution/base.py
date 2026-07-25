"""Execution backend and turn-scoped session protocols."""

from __future__ import annotations

from typing import Protocol

from chulk.execution.models import (
    CommandExecutionRequest,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSessionRequest,
    ExecutionWorkspace,
    FileListRequest,
    FileReadRequest,
    FileSearchRequest,
    FileWriteRequest,
    PatchApplyRequest,
)


class ExecutionSession(Protocol):
    """One coherent workspace used by all execution tools in a turn."""

    @property
    def workspace(self) -> ExecutionWorkspace: ...

    @property
    def policy(self) -> ExecutionPolicy: ...

    @property
    def closed(self) -> bool: ...

    def read_file(self, request: FileReadRequest) -> ExecutionResult: ...

    def write_file(self, request: FileWriteRequest) -> ExecutionResult: ...

    def apply_patch(self, request: PatchApplyRequest) -> ExecutionResult: ...

    def list_files(self, request: FileListRequest) -> ExecutionResult: ...

    def search_files(self, request: FileSearchRequest) -> ExecutionResult: ...

    def run_command(self, request: CommandExecutionRequest) -> ExecutionResult: ...

    async def read_file_async(self, request: FileReadRequest) -> ExecutionResult: ...

    async def write_file_async(self, request: FileWriteRequest) -> ExecutionResult: ...

    async def apply_patch_async(self, request: PatchApplyRequest) -> ExecutionResult: ...

    async def list_files_async(self, request: FileListRequest) -> ExecutionResult: ...

    async def search_files_async(self, request: FileSearchRequest) -> ExecutionResult: ...

    async def run_command_async(self, request: CommandExecutionRequest) -> ExecutionResult: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...


class ExecutionBackend(Protocol):
    """Host-selected factory for isolated or direct execution sessions."""

    @property
    def name(self) -> str: ...

    def open_session(self, request: ExecutionSessionRequest) -> ExecutionSession: ...

    async def open_session_async(self, request: ExecutionSessionRequest) -> ExecutionSession: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...
