"""Compatibility alias for the installed PCM WebSocket implementation."""

import sys

from whisper_runtime import pcm_live as _implementation

sys.modules[__name__] = _implementation
