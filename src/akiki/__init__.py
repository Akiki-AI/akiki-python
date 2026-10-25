"""Small, offline streaming voice activity detector."""
from .vad import SpeechFrame, VAD

__version__ = "0.0.5"
__all__ = ["SpeechFrame", "VAD", "__version__"]
