"""Offline streaming VAD with Rust, ONNX and PyTorch inference engines."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechFrame:
    start_seconds: float
    end_seconds: float
    probability: float
    speech: bool


class VAD:
    """Stateful 16 kHz mono VAD. Call reset() between recordings.

    ``engine="auto"`` tries local Rust, ONNX and PyTorch in that order.
    A named engine is required to succeed. ``library_path`` pins a Rust library.
    ``engine`` reports the selected engine; ``engine_errors`` explains fallbacks.
    With no ``model_precision`` specified, Rust uses INT8 and optional ONNX or
    PyTorch engines use FP32. An explicit precision is always honored.
    Converted weights are cached locally. FFT, recurrent state, biases and
    readout remain FP32.
    """

    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None,
                 *, engine: str = "auto", library_path: str | Path | None = None,
                 model_precision: str | None = None, fft_precision: str = "fp32",
                 cache_dir: str | Path | None = None):
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
        self._backend.reset()
        self.context = np.zeros(self.context_samples, dtype=np.float32)
        self.state = np.zeros((1, 1, self.hidden), dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.completed_samples = 0

    def push(self, samples, sample_rate: int = SAMPLE_RATE) -> list[SpeechFrame]:
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
        """Release this stream's engine resources. Safe to call repeatedly."""
        self._backend.close()
        if self.engine == "onnx":
            self.session = None

    def __enter__(self):
        self._backend.check_open()
        return self

    def __exit__(self, *_):
        self.close()
