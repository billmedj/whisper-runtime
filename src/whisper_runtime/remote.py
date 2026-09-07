"""Small CLI bridge to the installed live-v2 client; no native model imports."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping, Sequence
from threading import Event
from typing import Any
from urllib.parse import urlsplit

from .adapters.native_stream import StreamEventKind, TranscriptEvent
from .remote_transport import LiveConfig, stream_live


class RemoteError(RuntimeError):
    """Only fixed local codes reach CLI output; never transport exception text."""


def environment_headers(
    specifications: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    values = os.environ if environment is None else environment
    headers: dict[str, str] = {}
    names: set[str] = set()
    for specification in specifications:
        name, separator, variable = specification.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", variable)
        ):
            raise RemoteError("invalid_header_environment_mapping")
        lowered = name.lower()
        if (
            lowered in names
            or lowered in {"host", "connection", "upgrade", "content-length"}
            or lowered.startswith("sec-websocket-")
        ):
            raise RemoteError("duplicate_or_reserved_header")
        value = values.get(variable)
        if not value or "\r" in value or "\n" in value or "\x00" in value:
            raise RemoteError("missing_or_invalid_authentication_environment")
        headers[name] = value
        names.add(lowered)
    return headers


def validate_server(server: str) -> None:
    try:
        parsed = urlsplit(server)
        valid = (
            parsed.scheme in {"ws", "wss"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (
                parsed.scheme == "wss"
                or parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
        parsed.port  # Validate syntax without logging the supplied address.
    except ValueError:
        valid = False
    if not valid:
        raise RemoteError(
            "invalid_server_use_wss_or_loopback_ws_without_url_credentials"
        )


async def _source(
    chunks: Iterator[bytes], cancel: Event
) -> AsyncGenerator[bytes, None]:
    """Read the existing bounded source off-loop, only when live-v2 requests it."""
    pending: asyncio.Task[bytes | None] | None = None
    try:
        while not cancel.is_set():
            pending = asyncio.create_task(asyncio.to_thread(next, chunks, None))
            content = await asyncio.shield(pending)
            pending = None
            if cancel.is_set():
                raise asyncio.CancelledError
            if content is None:
                return
            yield content
        raise asyncio.CancelledError
    finally:
        cancel.set()
        # Cancellation must not close a generator while its next() is executing.
        if pending is not None:
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.2)


async def transcribe_remote(
    server: str,
    chunks: Iterator[bytes],
    *,
    cancel: Event,
    on_event: Callable[[TranscriptEvent], None],
    headers: Mapping[str, str] | None = None,
    config: LiveConfig | None = None,
) -> int:
    """Return verified completed sample count, never merely received FINAL."""
    validate_server(server)
    secrets = list((headers or {}).values())
    for name, value in (headers or {}).items():
        if name.lower() in {"authorization", "proxy-authorization"}:
            parts = value.split(None, 1)
            if len(parts) == 2:
                secrets.append(parts[1])
    source = _source(chunks, cancel)

    async def receive(value: dict[str, Any], elapsed_ns: int) -> None:
        del elapsed_ns
        text = value.get("text")
        # The transport redacts returned evidence, but its callbacks intentionally
        # retain raw SDK events. Refuse secret-reflecting text rather than edit a
        # committed revision or expose an authentication value in stdout/exports.
        if isinstance(text, str) and any(
            secret and secret in text for secret in (*secrets, server)
        ):
            raise RemoteError("sensitive_text_reflected_by_server")
        event = TranscriptEvent(**{**value, "kind": StreamEventKind(value["kind"])})
        await asyncio.to_thread(on_event, event)

    try:
        try:
            result = await stream_live(
                server, source, headers=headers, config=config, on_event=receive
            )
        finally:
            cancel.set()
            await source.aclose()
            if hasattr(chunks, "close"):
                chunks.close()
    except Exception:
        # Include bridge cleanup and callback conversion in the same sanitized
        # failure boundary as the transport. Never print arbitrary remote text.
        raise RemoteError("remote_client_failed") from None
    if result["status"] == "cancelled":
        raise KeyboardInterrupt
    if result["status"] != "completed":
        code = result.get("error_code")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]+", code):
            code = "remote_failed"
        raise RemoteError(code)
    samples = result.get("sample_count")
    if type(samples) is not int or samples <= 0:
        raise RemoteError("invalid_completed_sample_count")
    return samples
