"""Small Telegram Bot API client with sanitized failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import PurePath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chulk.gateway.commands import SHARED_CHANNEL_COMMANDS


TELEGRAM_API_BASE_URL = "https://api.telegram.org"
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    (item.name, item.description) for item in SHARED_CHANNEL_COMMANDS
) + (
    ("plan", "Prepare an approval plan"),
    ("approve", "Approve the pending plan"),
    ("reject", "Reject the pending plan"),
    ("help", "Show command help"),
)
TELEGRAM_SCHEDULING_COMMANDS: tuple[tuple[str, str], ...] = (
    ("reminders", "List scheduled tasks"),
    ("cancel", "Cancel a scheduled task"),
)
JsonRequest = Callable[[str, dict[str, object], float], object]
BinaryRequest = Callable[[str, float, int], bytes]
UploadRequest = Callable[
    [str, dict[str, object], str, bytes, str, str, float],
    object,
]


class TelegramError(RuntimeError):
    """A sanitized Telegram transport or protocol failure."""


@dataclass(frozen=True)
class TelegramAttachment:
    """One bounded Telegram file reference attached to a message."""

    file_id: str
    kind: str
    mime_type: str
    file_name: str | None = None


@dataclass(frozen=True)
class TelegramUpdate:
    """One supported text or media message extracted from an update."""

    update_id: int
    chat_id: int
    user_id: int
    text: str
    chat_type: str
    attachment: TelegramAttachment | None = None


class TelegramClient:
    """Call the subset of Telegram Bot API methods needed by the adapter."""

    def __init__(
        self,
        bot_token: str,
        *,
        request_json: JsonRequest | None = None,
        request_binary: BinaryRequest | None = None,
        request_upload: UploadRequest | None = None,
        request_timeout_seconds: float = 40.0,
    ) -> None:
        if not bot_token.strip():
            raise ValueError("bot_token is required")
        self._api_url = f"{TELEGRAM_API_BASE_URL}/bot{bot_token.strip()}"
        self._request_json = request_json or _request_json
        self._request_binary = request_binary or _request_binary
        self._request_upload = request_upload or _request_upload
        self._request_timeout_seconds = request_timeout_seconds
        self.next_offset: int | None = None
        self.ignored_updates: tuple[tuple[int, str], ...] = ()

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
        ignored_updates: list[tuple[int, str]] = []
        for item in result:
            update_id: int | None = None
            if isinstance(item, dict):
                raw_update_id = item.get("update_id")
                if isinstance(raw_update_id, int):
                    update_id = raw_update_id
                    self.next_offset = max(self.next_offset or 0, raw_update_id + 1)
            update = _parse_text_update(item)
            if update is not None:
                updates.append(update)
            elif update_id is not None:
                ignored_updates.append((update_id, _unsupported_destination(item)))
        self.ignored_updates = tuple(ignored_updates)
        return tuple(updates)

    def send_message(self, chat_id: int, text: str) -> None:
        """Send text, splitting it at Telegram's message boundary."""
        for part in split_message(text):
            self._call("sendMessage", {"chat_id": chat_id, "text": part})

    def download_file(self, file_id: str, *, max_bytes: int) -> bytes:
        """Resolve and download one Telegram file with a hard byte limit."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        result = self._call("getFile", {"file_id": file_id})
        if not isinstance(result, dict) or not isinstance(result.get("file_path"), str):
            raise TelegramError("Telegram getFile returned an invalid result")
        file_path = result["file_path"].lstrip("/")
        try:
            data = self._request_binary(
                f"{self._api_url.replace('/bot', '/file/bot', 1)}/{file_path}",
                self._request_timeout_seconds,
                max_bytes,
            )
        except Exception as exc:
            if isinstance(exc, TelegramError):
                raise
            raise TelegramError("Telegram file download failed") from exc
        if len(data) > max_bytes:
            raise TelegramError("Telegram attachment exceeds the configured size limit")
        return data

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show a short-lived activity indicator in one chat."""
        self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    def send_attachment(
        self,
        chat_id: int,
        data: bytes,
        *,
        mime_type: str,
        file_name: str,
        caption: str | None = None,
    ) -> None:
        """Upload one already-approved bounded attachment."""
        if not data:
            raise TelegramError("Telegram attachment cannot be empty")
        method, field = (
            ("sendPhoto", "photo")
            if mime_type.startswith("image/")
            else (
                ("sendAudio", "audio")
                if mime_type.startswith("audio/")
                else (
                    ("sendVideo", "video")
                    if mime_type.startswith("video/")
                    else ("sendDocument", "document")
                )
            )
        )
        payload: dict[str, object] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
        try:
            response = self._request_upload(
                f"{self._api_url}/{method}",
                payload,
                field,
                data,
                file_name,
                mime_type,
                self._request_timeout_seconds,
            )
        except Exception as exc:
            if isinstance(exc, TelegramError):
                raise
            raise TelegramError("Telegram attachment upload failed") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise TelegramError("Telegram attachment upload failed")

    def set_commands(
        self,
        commands: tuple[tuple[str, str], ...] = TELEGRAM_COMMANDS,
    ) -> None:
        """Register the adapter's command menu with Telegram."""
        command_payload = [
            {"command": command, "description": description}
            for command, description in commands
        ]
        self._call("setMyCommands", {"commands": command_payload})

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
    caption = message.get("caption")
    attachment = _parse_attachment(message)
    if not isinstance(chat, dict) or not isinstance(sender, dict):
        return None
    if not isinstance(text, str) and not isinstance(caption, str) and attachment is None:
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
        text=text if isinstance(text, str) else caption if isinstance(caption, str) else "",
        chat_type=chat_type,
        attachment=attachment,
    )


def _unsupported_destination(value: object) -> str:
    if not isinstance(value, dict):
        return "unknown"
    message = value.get("message")
    if not isinstance(message, dict):
        return "unknown"
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return "unknown"
    chat_id = chat.get("id")
    return str(chat_id) if isinstance(chat_id, int) else "unknown"


def _parse_attachment(message: dict[str, object]) -> TelegramAttachment | None:
    photo = message.get("photo")
    if isinstance(photo, list):
        candidates = [item for item in photo if isinstance(item, dict)]
        if candidates:
            selected = candidates[-1]
            file_id = selected.get("file_id")
            if isinstance(file_id, str):
                return TelegramAttachment(file_id, "image", "image/jpeg")
    for field, kind, default_mime in (
        ("voice", "voice", "audio/ogg"),
        ("audio", "audio", "audio/mpeg"),
        ("video", "video", "video/mp4"),
        ("video_note", "video", "video/mp4"),
        ("document", "document", "application/octet-stream"),
    ):
        value = message.get(field)
        if not isinstance(value, dict) or not isinstance(value.get("file_id"), str):
            continue
        mime = value.get("mime_type")
        raw_name = value.get("file_name")
        file_name = (
            PurePath(raw_name).name[:255]
            if isinstance(raw_name, str) and PurePath(raw_name).name
            else None
        )
        return TelegramAttachment(
            value["file_id"],
            kind,
            mime if isinstance(mime, str) else default_mime,
            file_name,
        )
    return None


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


def _request_binary(url: str, timeout: float, max_bytes: int) -> bytes:
    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise TelegramError("Telegram file download failed") from exc
    if len(data) > max_bytes:
        raise TelegramError("Telegram attachment exceeds the configured size limit")
    return data


def _request_upload(
    url: str,
    payload: dict[str, object],
    field_name: str,
    data: bytes,
    file_name: str,
    mime_type: str,
    timeout: float,
) -> object:
    boundary = f"chulk-{len(data):x}-{abs(hash(file_name)):x}"
    chunks: list[bytes] = []
    for key, value in payload.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            )
        )
    safe_name = PurePath(file_name).name.replace('"', "")
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{safe_name}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    request = Request(
        url,
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelegramError("Telegram attachment upload failed") from exc
