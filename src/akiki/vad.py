"""Offline streaming VAD with Rust, ONNX and PyTorch inference engines."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

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
    own ``chunk_samples`` value.

    Models load locally. ``engine="auto"`` tries Rust, ONNX Runtime, then
    PyTorch. The Rust engine uses INT8 by default; the optional runtimes use
    FP32. Explicit non-FP32 precision requires Rust. ONNX Runtime and PyTorch
    can be installed with ``akiki[onnx]`` and ``akiki[pytorch]`` respectively.
    See ``VAD.__init__`` for all constructor options.

    Use the detector as a context manager or call ``close()`` when finished.
    Closing does not pad or score a final incomplete frame.

    Attributes
    ----------
    engine : str
        Selected inference engine: ``"rust"``, ``"onnx"``, or ``"pytorch"``.
    engine_errors : dict[str, str]
        Reasons engines were unavailable before automatic selection succeeded.
    model_id : str
        Identifier from the loaded model's metadata.
    model_precision : str
        Active model precision, after resolving the engine's default.
    fft_precision : str
        FFT precision; always ``"fp32"``.
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

    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None,
                 *, engine: str = "auto", library_path: str | Path | None = None,
                 model_precision: str | None = None, fft_precision: str = "fp32",
                 cache_dir: str | Path | None = None):
        """Load a local model and inference engine, ready for a new stream.

        Parameters
        ----------
        model_path : str or pathlib.Path, optional
            Local model path, such as ``"vad.onnx"``. A JSON metadata file with
            the same stem is required. ONNX and PyTorch use matching ``.onnx``
            and ``.pt`` files; Rust uses the sibling ``<stem>-native/`` directory
            containing ``config.json``, ``model.safetensors``, and ``export.json``.
            If omitted, use the bundled model.
        threshold : float, optional
            Speech decision threshold between 0 and 1 inclusive. If omitted,
            use the threshold in the model's metadata. This changes the boolean
            speech decision, not the returned probability.
        engine : {"auto", "rust", "onnx", "pytorch"}, default="auto"
            Engine to use. Automatic selection tries Rust, ONNX Runtime, then
            PyTorch. A named engine must succeed; it does not fall back.
        library_path : str or pathlib.Path, optional
            Specific local Rust shared library to load. Affects only the Rust
            engine. If omitted, use ``AKIKI_VAD_LIBRARY`` or discover a bundled
            or locally built library.
        model_precision : str, optional
            One of ``"fp32"``, ``"fp16"``, ``"bf16"``, ``"int8"``, ``"int4"``,
            or ``"int8-fp32-input"``. If omitted, Rust uses INT8 and ONNX/PyTorch
            use FP32. Explicit non-FP32 precision requires Rust, including in
            automatic mode. Reduced-precision Rust weights are generated from
            the verified local FP32 model and reused from the cache.
            ``"int8-fp32-input"`` uses INT8 weights while keeping the first
            projection's input in FP32.
        fft_precision : {"fp32"}, default="fp32"
            FFT precision. Only FP32 is supported.
        cache_dir : str or pathlib.Path, optional
            Directory for generated Rust model variants. If omitted, use
            ``AKIKI_CACHE_DIR`` or the platform's user cache directory. Original
            model weights are preserved. FP32 inference needs no conversion.

        Raises
        ------
        ValueError
            A parameter, model specification, or model checksum is invalid.
        OSError
            The local model metadata cannot be read.
        RuntimeError
            No compatible engine is available for the requested configuration,
            or the directory for a generated model variant is not writable.
        """
        from ._engines import load_engine
        from ._quantization import MODEL_PRECISIONS

        if engine not in {"auto", "rust", "onnx", "pytorch"}:
            raise ValueError("Engine must be auto, rust, onnx or pytorch")
        if model_precision is not None and model_precision not in MODEL_PRECISIONS:
            raise ValueError("model_precision must be one of: " + ", ".join(MODEL_PRECISIONS))
        if fft_precision != "fp32":
            raise ValueError("fft_precision is fixed at fp32; integer and reduced-precision FFTs are not supported")
        if model_precision not in {None, "fp32"} and engine not in {"auto", "rust"}:
            raise ValueError("Non-FP32 model_precision requires engine='rust' or engine='auto'")
        self.fft_precision = fft_precision
        path = Path(model_path) if model_path is not None else Path(__file__).parent / "models/akiki-vad.onnx"
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.model_id = metadata.get("model_id", path.stem)
        spec = metadata["spec"]
        self.chunk_samples = int(spec["chunk_samples"])
        self.context_samples = int(spec["context_samples"])
        self.hidden = int(spec["hidden"])
        if spec["kind"] != "gru" or (self.chunk_samples, self.context_samples) not in {(256, 320), (512, 64)}:
            raise ValueError("Expected an Akiki full-waveform GRU model")
        if spec.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE or self.hidden <= 0:
            raise ValueError("Expected a 16 kHz model with a positive hidden size")
        self.threshold = float(metadata["threshold"] if threshold is None else threshold)
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must lie between zero and one")
        self._backend, self.engine, self.engine_errors = load_engine(
            path, metadata, engine, library_path, model_precision, cache_dir)
        self.model_precision = model_precision or ("int8" if self.engine == "rust" else "fp32")
        if self.engine == "onnx":
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
            window = np.concatenate([self.context, audio[start:start + self.chunk_samples]])
            probability, state = self._backend.infer(window, self.state)
            probability = float(np.asarray(probability).reshape(-1)[0])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError("Model returned an invalid speech probability")
            begin = self.completed_samples
            self.state = state
            self.context = window[-self.context_samples:].copy()
            self.completed_samples += self.chunk_samples
            frames.append(SpeechFrame(begin / SAMPLE_RATE, self.completed_samples / SAMPLE_RATE,
                                      probability, probability >= self.threshold))
        self.pending = audio[complete:].copy()
        return frames

    def close(self) -> None:
        """Release this stream's inference resources; safe to call repeatedly.

        Pending samples are not scored. After closing, ``push()``, ``reset()``,
        and entering a ``with`` block raise ``RuntimeError``. Previously
        returned ``SpeechFrame`` objects remain usable.
        """
        self._backend.close()
        if self.engine == "onnx":
            self.session = None

    def __enter__(self):
        """Return this open detector for use in a ``with`` block."""
        self._backend.check_open()
        return self

    def __exit__(self, *_):
        """Close the detector on leaving a ``with`` block; propagate exceptions."""
        self.close()
