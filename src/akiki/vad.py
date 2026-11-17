"""Streaming speech detection and its configuration and result types.

Use ``VAD`` with successive mono, 16 kHz audio buffers. Each complete frame
returns a ``SpeechFrame`` with timestamps, a probability, and a speech decision.
The bundled model processes 256 samples (16 ms) per frame.

The Akiki engine is the primary engine. ``Engine.AUTO`` tries it first.
The ONNX and PyTorch engines provide compatibility fallbacks.
``ModelPrecision`` selects the VAD model's numeric format; ``FFTPrecision`` configures its FFT.
These enums belong to VAD; MiniLM uses ``akiki.decision.ModelPrecision``.

Examples
--------
>>> from akiki import vad
>>> with vad.VAD(model_precision=vad.ModelPrecision.FP32) as detector:
...     frames = detector.push([0.0] * 256)
>>> len(frames)
1
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._engines import Engine
from ._precision import FFTPrecision, ModelPrecision

__all__ = ["Engine", "FFTPrecision", "ModelPrecision", "SpeechFrame", "VAD"]

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechFrame:
    """Immutable speech decision for one complete frame returned by ``VAD.push``.

    Timestamps describe the input audio, measured from detector creation or the
    last ``VAD.reset()`` call. They do not measure processing or wall-clock time.
    The bundled model produces one frame for every 256 samples (16 ms).

    Attributes
    ----------
    start_seconds : float
        Inclusive start time of the frame, in seconds.
    end_seconds : float
        Exclusive end time of the frame, in seconds.
    probability : float
        Model's speech probability, between 0 and 1 inclusive.
    speech : bool
        Whether ``probability >= detector.threshold`` when the frame was
        produced. Decisions are made per frame, without merging speech segments.

    Examples
    --------
    >>> frame = SpeechFrame(0.0, 0.016, 0.9, True)
    >>> frame.end_seconds - frame.start_seconds
    0.016
    >>> frame.probability, frame.speech
    (0.9, True)
    """

    start_seconds: float
    end_seconds: float
    probability: float
    speech: bool


class VAD:
    """Identify speech in a stream of mono audio sampled at 16,000 Hz.

    Each instance maintains the recurrent state, audio context, and incomplete
    frame for one stream. Feed successive buffers to ``push()``; call
    ``reset()`` before an independent recording. The bundled model emits a
    ``SpeechFrame`` for every 256 samples (16 ms). Custom models use their
    supported ``chunk_samples`` value: 256 or 512 samples.

    Models load locally. The Akiki engine is the primary engine.
    ``engine=Engine.AUTO`` tries it first, then the ONNX and PyTorch compatibility
    fallbacks. The Akiki engine uses INT8 by default. Compatibility engines use FP32.
    Explicit non-FP32 precision requires the Akiki engine. Install compatibility
    runtimes with ``akiki[onnx]`` or ``akiki[pytorch]`` when needed.
    See ``VAD.__init__`` for all constructor options.

    Use the detector as a context manager or call ``close()`` when finished.
    Closing does not pad or score a final incomplete frame.

    Attributes
    ----------
    engine : Engine
        Selected inference engine: ``Engine.AKIKI``, ``Engine.ONNX``, or
        ``Engine.PYTORCH``.
    engine_errors : dict[Engine, str]
        Reasons engines were unavailable before automatic selection succeeded.
    model_id : str
        Identifier from the loaded model's metadata.
    model_precision : ModelPrecision
        Active model precision, after resolving the engine's default.
    fft_precision : FFTPrecision
        FFT precision; always ``FFTPrecision.FP32``.
    threshold : float
        Minimum probability classified as speech.
    chunk_samples : int
        Number of input samples needed for one output frame.
    context_samples : int
        Number of prior audio samples retained as model context.
    hidden : int
        Number of recurrent hidden units in the loaded model.
    completed_samples : int
        Number of samples scored since creation or the last reset.
    pending : numpy.ndarray
        Unscored float32 samples buffered for the next call to ``push``.
    context : numpy.ndarray
        Float32 audio context with shape ``(context_samples,)``.
    state : numpy.ndarray
        Float32 recurrent state with shape ``(1, 1, hidden)``.
    session : onnxruntime.InferenceSession
        Present only when the selected engine is ONNX; ``None`` after closing.

    Notes
    -----
    Treat ``pending``, ``context``, and ``state`` as read-only stream state.
    Use a separate instance for each independent audio stream.
    Calls on one detector are not protected by a lock. Do not call ``push``,
    ``reset``, or ``close`` concurrently on the same instance.

    Examples
    --------
    Successive buffers share a timeline and incomplete frames are retained::

        >>> from akiki import VAD
        >>> with VAD() as detector:
        ...     first = detector.push([0.0] * 100)
        ...     second = detector.push([0.0] * 156)
        >>> first
        []
        >>> len(second), second[0].end_seconds
        (1, 0.016)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float | None = None,
        *,
        engine: Engine = Engine.AUTO,
        library_path: str | Path | None = None,
        model_precision: ModelPrecision | None = None,
        fft_precision: FFTPrecision = FFTPrecision.FP32,
        cache_dir: str | Path | None = None,
    ):
        """Load a local model and inference engine, ready for a new stream.

        Parameters
        ----------
        model_path : str or pathlib.Path, optional
            If omitted, use the bundled model.
            For a custom Akiki model, pass its JSON metadata path, such as
            ``"models/vad.json"``. The Akiki engine loads the sibling
            ``"models/vad-native/"`` directory. It must contain ``config.json``,
            ``model.safetensors``, and ``export.json``.
            The ONNX and PyTorch compatibility fallbacks use matching ``.onnx``
            and ``.pt`` files beside the metadata file.
            Custom models must use the full-waveform GRU format. Supported
            ``(chunk_samples, context_samples)`` pairs are ``(256, 320)`` and
            ``(512, 64)``. Custom PyTorch metadata also requires ``pytorch_sha256``.
        threshold : float, optional
            Speech decision threshold between 0 and 1 inclusive. If omitted,
            use the threshold in the model's metadata. This changes the boolean
            speech decision, not the returned probability.
        engine : Engine, default=Engine.AUTO
            Automatic selection tries the primary Akiki engine first.
            ONNX and PyTorch are compatibility fallbacks, in that order.
            An explicitly selected engine must succeed; it does not fall back.
            Pass an ``Engine`` member, such as ``Engine.AKIKI``.
        library_path : str or pathlib.Path, optional
            Specific local Akiki shared library to load. Affects only the Akiki
            engine. If omitted, use ``AKIKI_VAD_LIBRARY`` or the bundled library.
            Automatic discovery does not search developer directories.
        model_precision : ModelPrecision, optional
            One of ``ModelPrecision.FP32``, ``FP16``, ``BF16``, ``INT8``,
            ``INT4``, or ``INT8_FP32_INPUT``. If omitted, Akiki uses INT8 and
            ONNX/PyTorch use FP32. Explicit non-FP32 precision requires Akiki,
            including in automatic mode. Reduced-precision Akiki weights are
            generated from the verified local FP32 model and reused from the
            cache. ``ModelPrecision.INT8_FP32_INPUT`` uses INT8 weights while
            keeping the first projection's input in FP32.
        fft_precision : FFTPrecision, default=FFTPrecision.FP32
            FFT precision. Only ``FFTPrecision.FP32`` is supported.
        cache_dir : str or pathlib.Path, optional
            Directory for generated Akiki model variants. If omitted, use
            ``AKIKI_CACHE_DIR`` or the platform's user cache directory. Original
            model weights are preserved. FP32 inference needs no conversion.

        Raises
        ------
        TypeError
            ``engine``, ``model_precision``, or ``fft_precision`` has the wrong
            enum type. ``model_precision`` also accepts ``None``.
        ValueError
            A parameter, model specification, or model checksum is invalid.
        OSError
            The local model metadata cannot be read.
        RuntimeError
            No compatible engine is available for the requested configuration,
            or the directory for a generated model variant is not writable.
        """
        from ._engines import load_engine

        if not isinstance(engine, Engine):
            raise TypeError("engine must be an Engine member, such as Engine.AUTO or Engine.AKIKI")
        if model_precision is not None and not isinstance(model_precision, ModelPrecision):
            raise TypeError("model_precision must be a ModelPrecision member or None")
        if not isinstance(fft_precision, FFTPrecision):
            raise TypeError("fft_precision must be an FFTPrecision member")
        if model_precision not in (None, ModelPrecision.FP32) and engine not in (
            Engine.AUTO,
            Engine.AKIKI,
        ):
            raise ValueError("Non-FP32 model_precision requires Engine.AKIKI or Engine.AUTO")
        self.fft_precision = fft_precision
        path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).parent / "models/akiki-vad.onnx"
        )
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.model_id = metadata.get("model_id", path.stem)
        spec = metadata["spec"]
        self.chunk_samples = int(spec["chunk_samples"])
        self.context_samples = int(spec["context_samples"])
        self.hidden = int(spec["hidden"])
        if spec["kind"] != "gru" or (self.chunk_samples, self.context_samples) not in {
            (256, 320),
            (512, 64),
        }:
            raise ValueError("Expected an Akiki full-waveform GRU model")
        if spec.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE or self.hidden <= 0:
            raise ValueError("Expected a 16 kHz model with a positive hidden size")
        self.threshold = float(metadata["threshold"] if threshold is None else threshold)
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must lie between zero and one")
        self._backend, self.engine, self.engine_errors = load_engine(
            path, metadata, engine, library_path, model_precision, cache_dir
        )
        self.model_precision = model_precision or (
            ModelPrecision.INT8 if self.engine is Engine.AKIKI else ModelPrecision.FP32
        )
        if self.engine is Engine.ONNX:
            self.session = self._backend.session
        self.reset()

    def reset(self) -> None:
        """Start an independent recording with the same model and settings.

        Clear the recurrent state and audio context, discard pending samples,
        and reset output timestamps to zero. The model remains loaded.

        Raises
        ------
        RuntimeError
            The detector has been closed. Create a new instance to process more
            audio after closing.
        """
        self._backend.reset()
        self.context = np.zeros(self.context_samples, dtype=np.float32)
        self.state = np.zeros((1, 1, self.hidden), dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.completed_samples = 0

    def push(self, samples, sample_rate: int = SAMPLE_RATE) -> list[SpeechFrame]:
        """Process the next audio buffer and return decisions for complete frames.

        Parameters
        ----------
        samples : array_like
            One-dimensional, finite mono waveform samples. Lists and NumPy
            arrays are accepted and converted to float32. Samples should
            normally be scaled to [-1, 1]; integer PCM is not normalized for
            you. Audio is not resampled or mixed down from multiple channels.
        sample_rate : int, default=16000
            Input sample rate in Hz. Must be 16000.

        Returns
        -------
        list[SpeechFrame]
            One result for each complete ``chunk_samples`` input samples,
            including any samples buffered by a previous call. Timestamps
            continue from earlier calls. Returns an empty list when there is
            not enough audio for a complete frame.

        Raises
        ------
        TypeError
            The input type cannot be converted to numeric audio samples.
        ValueError
            Audio cannot be converted to float32, is not one-dimensional,
            contains nonfinite values, or has the wrong sample rate. These
            validation failures leave the stream state unchanged.
        RuntimeError
            The detector is closed, inference fails, or the model returns an
            invalid speech probability.

        Notes
        -----
        Buffers may have any length, including zero. A final incomplete frame
        stays in ``pending`` until more audio arrives or ``reset()`` discards
        it. Closing the detector does not pad or score those samples.
        """
        self._backend.check_open()
        audio = np.asarray(samples, dtype=np.float32)
        if sample_rate != SAMPLE_RATE or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono 16 kHz audio samples")
        audio = np.concatenate([self.pending, audio])
        complete = len(audio) // self.chunk_samples * self.chunk_samples
        frames = []
        for start in range(0, complete, self.chunk_samples):
            window = np.concatenate([self.context, audio[start : start + self.chunk_samples]])
            probability, state = self._backend.infer(window, self.state)
            probability = float(np.asarray(probability).reshape(-1)[0])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError("Model returned an invalid speech probability")
            begin = self.completed_samples
            self.state = state
            self.context = window[-self.context_samples :].copy()
            self.completed_samples += self.chunk_samples
            frames.append(
                SpeechFrame(
                    begin / SAMPLE_RATE,
                    self.completed_samples / SAMPLE_RATE,
                    probability,
                    probability >= self.threshold,
                )
            )
        self.pending = audio[complete:].copy()
        return frames

    def close(self) -> None:
        """Release this stream's inference resources; safe to call repeatedly.

        Pending samples are not scored. After closing, ``push()``, ``reset()``,
        and entering a ``with`` block raise ``RuntimeError``. Previously
        returned ``SpeechFrame`` objects remain usable.
        """
        self._backend.close()
        if self.engine is Engine.ONNX:
            self.session = None

    def __enter__(self):
        """Return this open detector for use in a ``with`` block."""
        self._backend.check_open()
        return self

    def __exit__(self, *_):
        """Close the detector on leaving a ``with`` block; propagate exceptions."""
        self.close()
