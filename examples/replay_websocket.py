"""Compatibility alias for the installed PCM WebSocket implementation."""

import sys

from whisper_runtime import remote_transport as _implementation

sys.modules[__name__] = _implementation
