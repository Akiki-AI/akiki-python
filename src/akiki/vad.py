"""Deprecated imports for VAD. Use ``akiki.audio.vad``."""

import sys
import warnings

from .audio.vad import SAMPLE_RATE as SAMPLE_RATE
from .audio.vad import VAD as VAD
from .audio.vad import Engine as Engine
from .audio.vad import FFTPrecision as FFTPrecision
from .audio.vad import ModelPrecision as ModelPrecision
from .audio.vad import SpeechFrame as SpeechFrame


def _deprecation_stacklevel() -> int:
    # The warnings module skips the importlib bootstrap frames itself.
    # Also skip the Python import wrapper and our package's lazy import.
    level = 2
    frame = sys._getframe(1).f_back
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module in (__package__, "importlib"):
            level += 1
        elif not module.startswith("importlib._bootstrap"):
            break
        frame = frame.f_back
    return level


warnings.warn(
    "akiki.vad is deprecated and will be removed in version 0.3.0. "
    "Use 'from akiki.audio import vad' instead.",
    DeprecationWarning,
    stacklevel=_deprecation_stacklevel(),
)

__all__ = ["Engine", "FFTPrecision", "ModelPrecision", "SpeechFrame", "VAD"]
