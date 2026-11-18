"""Compatibility imports for voice activity detection in ``akiki.audio.vad``."""

from .audio.vad import SAMPLE_RATE as SAMPLE_RATE
from .audio.vad import VAD as VAD
from .audio.vad import Engine as Engine
from .audio.vad import FFTPrecision as FFTPrecision
from .audio.vad import ModelPrecision as ModelPrecision
from .audio.vad import SpeechFrame as SpeechFrame

__all__ = ["Engine", "FFTPrecision", "ModelPrecision", "SpeechFrame", "VAD"]
