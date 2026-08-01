"""Async cleanup helpers shared by core execution paths."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable


async def await_cleanup_after_error(
    cleanup: Awaitable[object],
    original: BaseException,
) -> None:
    """Complete required cleanup without replacing the triggering exception."""

    if not isinstance(original, asyncio.CancelledError):
        try:
            await cleanup
        except BaseException as cleanup_error:
            original.add_note(
                "async cleanup also failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        return

    task = asyncio.ensure_future(cleanup)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException as cleanup_error:
            original.add_note(
                "async cleanup also failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
    except BaseException as cleanup_error:
        original.add_note(
            "async cleanup also failed with "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
