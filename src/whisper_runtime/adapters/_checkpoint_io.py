"""Private, bounded JSON storage for whitelisted logical stream savepoints.

The checksum detects corruption, not malicious modification or provenance. No
pickle, class imports, model state, or arbitrary object construction is supported.
Publication never replaces a destination. File contents are synced before the
hard-link publication; POSIX also syncs the directory. Windows has no promised
power-loss durability. On Windows, callers must keep ancestor directories stable
against concurrent rename/reparse-point changes during filesystem operations.
"""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import hmac
import json
import math
import os
import stat
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

from .continuous_stream import ContinuousStreamConfig

_MAX_BYTES = 16 * 1024 * 1024
_MAX_DEPTH = 32
_SCHEMA = "continuous-savepoint/v1"


def _registry(classes: tuple[type, ...]) -> dict[str, type]:
    result: dict[str, type] = {}
    for cls in classes:
        if not isinstance(cls, type) or not is_dataclass(cls):
            raise ValueError("checkpoint registry requires dataclass types")
        if cls.__name__ in result or any(not field.init for field in fields(cls)):
            raise ValueError("duplicate class name or non-init dataclass field")
        result[cls.__name__] = cls
    return result


def _matches(value: object, annotation: object) -> bool:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is object:
        return True  # The wire codec still rejects unsupported runtime types.
    if origin in (Union, types.UnionType):
        return any(_matches(value, item) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(item, args[0]) for item in value)
        return len(value) == len(args) and all(
            _matches(item, hint) for item, hint in zip(value, args)
        )
    if origin is dict:
        return (
            isinstance(value, dict)
            and len(args) == 2
            and all(
                _matches(key, args[0]) and _matches(item, args[1])
                for key, item in value.items()
            )
        )
    if annotation in (bool, int, str, bytes, type(None)):
        return type(value) is annotation
    if annotation is float:
        return type(value) in (int, float)
    return (
        isinstance(annotation, type)
        and is_dataclass(annotation)
        and isinstance(value, annotation)
    )


def _validate_fields(cls: type, values: dict[str, object]) -> None:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(f"checkpoint fields do not match {cls.__name__}")
    hints = get_type_hints(cls)
    for name, value in values.items():
        if name not in hints or not _matches(value, hints[name]):
            raise ValueError(f"invalid checkpoint field {cls.__name__}.{name}")


def _pack(value: object, registry: dict[str, type], depth: int = 0) -> object:
    if depth > _MAX_DEPTH:
        raise ValueError("checkpoint exceeds maximum nesting")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str:
        if len(value) > _MAX_BYTES:
            raise ValueError("checkpoint string exceeds size limit")
        return value
    if type(value) is bytes:
        binary = value
        if len(binary) > _MAX_BYTES * 3 // 4:
            raise ValueError("checkpoint bytes exceed size limit")
        return {"tag": "bytes", "data": base64.b64encode(binary).decode("ascii")}
    if type(value) is tuple:
        return {
            "tag": "tuple",
            "items": [
                _pack(item, registry, depth + 1)
                for item in cast(tuple[object, ...], value)
            ],
        }
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if not all(type(key) is str for key in mapping):
            raise ValueError("checkpoint dictionary keys must be strings")
        return {
            "tag": "dict",
            "items": {
                key: _pack(item, registry, depth + 1) for key, item in mapping.items()
            },
        }
    cls = type(value)
    if registry.get(cls.__name__) is cls and is_dataclass(value):
        values = {field.name: getattr(value, field.name) for field in fields(value)}
        _validate_fields(cls, values)
        return {
            "tag": "dataclass",
            "class": cls.__name__,
            "fields": {
                name: _pack(item, registry, depth + 1) for name, item in values.items()
            },
        }
    raise ValueError(f"unsupported checkpoint type: {cls.__name__}")


def _unpack(value: object, registry: dict[str, type], depth: int = 0) -> object:
    if depth > _MAX_DEPTH:
        raise ValueError("checkpoint exceeds maximum nesting")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid checkpoint value")
    tag = value.get("tag")
    if tag == "bytes" and set(value) == {"tag", "data"}:
        data = value["data"]
        if not isinstance(data, str) or len(data) > _MAX_BYTES:
            raise ValueError("invalid checkpoint bytes")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid checkpoint base64") from exc
        if base64.b64encode(decoded).decode("ascii") != data:
            raise ValueError("non-canonical checkpoint base64")
        return decoded
    if tag == "tuple" and set(value) == {"tag", "items"}:
        if not isinstance(value["items"], list):
            raise ValueError("invalid checkpoint tuple")
        return tuple(_unpack(item, registry, depth + 1) for item in value["items"])
    if tag == "dict" and set(value) == {"tag", "items"}:
        if not isinstance(value["items"], dict):
            raise ValueError("invalid checkpoint dictionary")
        return {
            name: _unpack(item, registry, depth + 1)
            for name, item in value["items"].items()
        }
    if tag == "dataclass" and set(value) == {"tag", "class", "fields"}:
        name, raw_fields = value["class"], value["fields"]
        if (
            not isinstance(name, str)
            or name not in registry
            or not isinstance(raw_fields, dict)
        ):
            raise ValueError("unknown checkpoint class or invalid fields")
        cls = registry[name]
        values = {
            key: _unpack(item, registry, depth + 1) for key, item in raw_fields.items()
        }
        # These opt-ins did not exist in earlier v1 savepoints. Preserve the old
        # behavior; never infer other missing fields or accept unknown fields.
        if cls is ContinuousStreamConfig:
            missing = {field.name for field in fields(cls)} - values.keys()
            defaults = {
                "eof_context_retry": False,
                "defer_word_commits": False,
                "max_draft_tokens": 0,
                "previous_holdback_ms": 0,
            }
            if missing <= defaults.keys():
                values.update({name: defaults[name] for name in missing})
        _validate_fields(cls, values)
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint {name}") from exc
    raise ValueError("unknown checkpoint tag or unexpected fields")


def _canonical(value: object) -> bytes:
    output = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        encoded = chunk.encode("utf-8")
        if len(output) + len(encoded) > _MAX_BYTES:
            raise ValueError("checkpoint exceeds 16 MiB size limit")
        output.extend(encoded)
    return bytes(output)


def _check_raw(raw: bytes) -> None:
    if type(raw) is not bytes or len(raw) > _MAX_BYTES:
        raise ValueError("checkpoint must be bytes of at most 16 MiB")


def _check_json_depth(raw: bytes) -> None:
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > _MAX_DEPTH:
                raise ValueError("checkpoint JSON exceeds maximum nesting")
        elif char in (93, 125):
            depth -= 1


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite checkpoint number: {value}")


def encode(payload: dict[str, object], registry: tuple[type, ...]) -> bytes:
    """Validate and encode a fixed-whitelist logical snapshot, within 16 MiB."""
    if type(payload) is not dict:
        raise ValueError("checkpoint payload must be a dictionary")
    packed = _pack(payload, _registry(registry))
    digest = hashlib.sha256(_canonical(packed)).hexdigest()
    raw = _canonical({"schema": _SCHEMA, "sha256": digest, "payload": packed})
    _check_json_depth(raw)
    return raw


def decode(raw: bytes, registry: tuple[type, ...]) -> dict[str, object]:
    """Verify and reconstruct known dataclasses; never performs disk writes."""
    _check_raw(raw)
    _check_json_depth(raw)
    classes = _registry(registry)
    try:
        envelope = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid checkpoint JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "sha256",
        "payload",
    }:
        raise ValueError("invalid checkpoint envelope")
    digest = envelope["sha256"]
    if (
        envelope["schema"] != _SCHEMA
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("unsupported checkpoint schema or digest")
    actual = hashlib.sha256(_canonical(envelope["payload"])).hexdigest()
    if not hmac.compare_digest(digest, actual):
        raise ValueError("checkpoint checksum mismatch")
    payload = _unpack(envelope["payload"], classes)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return cast(dict[str, object], payload)


def _reject_special(info: os.stat_result, *, directory: bool = False) -> None:
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise OSError(
            errno.ELOOP, "checkpoint paths cannot contain links/reparse points"
        )
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise OSError(errno.EINVAL, "checkpoint path is not a regular file/directory")


def _check_ancestors(path: Path) -> None:
    for parent in reversed(path.parents):
        _reject_special(parent.lstat(), directory=True)


@contextmanager
def _location(path: str | Path) -> Iterator[tuple[Path, int | None]]:
    target = Path(path).absolute()
    if not target.name:
        raise ValueError("checkpoint requires a file path")
    if os.name == "nt":
        _check_ancestors(target)
        yield target, None
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target.anchor, flags)
    try:
        for component in target.parent.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield target, descriptor
    finally:
        os.close(descriptor)


def _name(path: Path, directory: int | None) -> str:
    return str(path) if directory is None else path.name


def write_new(path: str | Path, raw: bytes) -> None:
    """Atomically publish complete bytes without replacing an existing path."""
    _check_raw(raw)
    with _location(path) as (target, directory):
        destination = _name(target, directory)
        try:
            os.stat(destination, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                errno.EEXIST, "checkpoint already exists", str(target)
            )
        temporary = _name(
            target.with_name(f".savepoint-{uuid.uuid4().hex}.tmp"), directory
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if directory is None:
                _check_ancestors(target)
            os.link(
                temporary,
                destination,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        finally:
            os.unlink(temporary, dir_fd=directory)
        if directory is not None:
            os.fsync(directory)


def read(path: str | Path) -> bytes:
    """Read at most 16 MiB from a regular, non-symlink checkpoint file."""
    with _location(path) as (target, directory):
        name = _name(target, directory)
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        _reject_special(before)
        if before.st_size > _MAX_BYTES:
            raise ValueError("checkpoint exceeds 16 MiB size limit")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(name, flags, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            _reject_special(opened)
            if not os.path.samestat(before, opened):
                raise OSError(errno.EAGAIN, "checkpoint changed while opening")
            if directory is None:
                _check_ancestors(target)
            if opened.st_size > _MAX_BYTES:
                raise ValueError("checkpoint exceeds 16 MiB size limit")
            raw = stream.read(_MAX_BYTES + 1)
    _check_raw(raw)
    return raw
