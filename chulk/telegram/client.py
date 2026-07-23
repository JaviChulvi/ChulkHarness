"""Small Telegram Bot API client with sanitized failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TELEGRAM_API_BASE_URL = "https://api.telegram.org"
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_COMMANDS: tuple[tuple[str, str], ...] = (
    ("new", "Start a new conversation"),
    ("status", "Show provider, model, and conversation"),
    ("plan", "Prepare an approval plan"),
    ("approve", "Approve the pending plan"),
    ("reject", "Reject the pending plan"),
    ("help", "Show command help"),
)
JsonRequest = Callable[[str, dict[str, object], float], object]


class TelegramError(RuntimeError):
    """A sanitized Telegram transport or protocol failure."""


@dataclass(frozen=True)
class TelegramUpdate:
    """One supported text message extracted from a Telegram update."""

    update_id: int
    chat_id: int
    user_id: int
    text: str
    chat_type: str


class TelegramClient:
    """Call the subset of Telegram Bot API methods needed by the adapter."""

    def __init__(
        self,
        bot_token: str,
        *,
        request_json: JsonRequest | None = None,
        request_timeout_seconds: float = 40.0,
    ) -> None:
        if not bot_token.strip():
            raise ValueError("bot_token is required")
        self._api_url = f"{TELEGRAM_API_BASE_URL}/bot{bot_token.strip()}"
        self._request_json = request_json or _request_json
        self._request_timeout_seconds = request_timeout_seconds
        self.next_offset: int | None = None

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> tuple[TelegramUpdate, ...]:
        """Long-poll for supported text updates."""
        payload: dict[str, object] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call(
            "getUpdates",
            payload,
            timeout=max(self._request_timeout_seconds, timeout_seconds + 5.0),
        )
        if not isinstance(result, list):
            raise TelegramError("Telegram getUpdates returned an invalid result")
        updates: list[TelegramUpdate] = []
        for item in result:
            if isinstance(item, dict):
                update_id = item.get("update_id")
                if isinstance(update_id, int):
                    self.next_offset = max(self.next_offset or 0, update_id + 1)
            update = _parse_text_update(item)
            if update is not None:
                updates.append(update)
        return tuple(updates)

    def send_message(self, chat_id: int, text: str) -> None:
        """Send text, splitting it at Telegram's message boundary."""
        for part in split_message(text):
            self._call("sendMessage", {"chat_id": chat_id, "text": part})

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show a short-lived activity indicator in one chat."""
        self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    def set_commands(self) -> None:
        """Register the adapter's command menu with Telegram."""
        commands = [
            {"command": command, "description": description}
            for command, description in TELEGRAM_COMMANDS
        ]
        self._call("setMyCommands", {"commands": commands})

    def _call(
        self,
        method: str,
        payload: dict[str, object],
        *,
        timeout: float | None = None,
    ) -> object:
        try:
            response = self._request_json(
                f"{self._api_url}/{method}",
                payload,
                timeout or self._request_timeout_seconds,
            )
        except Exception as exc:
            if isinstance(exc, TelegramError):
                raise
            raise TelegramError(f"Telegram {method} request failed") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            description = response.get("description") if isinstance(response, dict) else None
            detail = f": {description}" if isinstance(description, str) else ""
            raise TelegramError(f"Telegram {method} failed{detail}")
        return response.get("result")


def split_message(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> tuple[str, ...]:
    """Split text into non-empty Telegram-sized chunks, preferring newlines."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    remaining = text.strip() or "(empty response)"
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def _parse_text_update(value: object) -> TelegramUpdate | None:
    if not isinstance(value, dict):
        return None
    update_id = value.get("update_id")
    message = value.get("message")
    if not isinstance(update_id, int) or not isinstance(message, dict):
        return None
    chat = message.get("chat")
    sender = message.get("from")
    text = message.get("text")
    if not isinstance(chat, dict) or not isinstance(sender, dict) or not isinstance(text, str):
        return None
    chat_id = chat.get("id")
    chat_type = chat.get("type")
    user_id = sender.get("id")
    if (
        not isinstance(chat_id, int)
        or not isinstance(user_id, int)
        or not isinstance(chat_type, str)
    ):
        return None
    return TelegramUpdate(
        update_id=update_id,
        chat_id=chat_id,
        user_id=user_id,
        text=text,
        chat_type=chat_type,
    )


def _request_json(url: str, payload: dict[str, object], timeout: float) -> object:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelegramError("Telegram HTTP request failed") from exc
