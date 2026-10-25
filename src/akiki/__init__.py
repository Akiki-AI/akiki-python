"""Detect speech locally in a stream of mono, 16 kHz audio samples.

Use ``VAD`` to process successive audio buffers and ``SpeechFrame``
to inspect each frame's timestamps, speech probability, and speech decision.
The bundled model processes 256 samples (16 ms) per frame. Model files are
included in the package; inference never downloads a model.

Examples
--------
Process one frame of silence and release the detector's resources on exit::

    >>> import akiki
    >>> with akiki.VAD() as detector:
    ...     frames = detector.push([0.0] * 256)
    >>> len(frames)
    1
    >>> frames[0].start_seconds, frames[0].end_seconds
    (0.0, 0.016)

For live audio, call ``detector.push(samples)`` repeatedly inside the ``with``
block. Incomplete frames are buffered for the next call. Call
``detector.reset()`` before starting an independent recording.

Use ``help(akiki.VAD)`` for configuration and engine requirements,
``help(akiki.VAD.push)`` for audio input rules, and ``help(akiki.SpeechFrame)``
for the result fields.

Attributes
----------
__version__ : str
    Installed Akiki package version.
"""
from .vad import SpeechFrame, VAD

__version__ = "0.0.5"
__all__ = ["SpeechFrame", "VAD", "__version__"]
