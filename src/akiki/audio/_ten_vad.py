"""TEN-VAD inference with the Akiki audio frontend."""

import ctypes
import weakref
from pathlib import Path

import numpy as np

from .._engines import (
    Backend,
    EngineUnavailable,
    _create_onnx_session,
    _require_onnx_runtime,
    library_candidates,
)


class TenVadBackend(Backend):
    chunk_samples = 256
    hidden = 64
    state_shape = (4, hidden)

    @classmethod
    def load_library(cls, library_path: str | Path | None):
        candidate = next(
            library_candidates(library_path, "akiki_ten_vad_engine", "AKIKI_TEN_VAD_LIBRARY")
        ).resolve()
        try:
            lib = ctypes.CDLL(str(candidate))
            cls._bind(lib)
        except (OSError, AttributeError) as error:
            raise EngineUnavailable(
                f"No compatible Akiki TEN-VAD library at {candidate}. "
                "Set AKIKI_TEN_VAD_LIBRARY or pass library_path."
            ) from error
        return lib

    def __init__(self, path: Path, lib):
        handle = lib.akiki_ten_vad_create(str(path.resolve()).encode("utf-8"))
        if not handle:
            raise ValueError(f"The Akiki TEN-VAD engine cannot load the model: {path}")
        self.lib, self.handle = lib, handle
        self._finalizer = weakref.finalize(self, lib.akiki_ten_vad_free, handle)

    @staticmethod
    def _bind(lib):
        pointer = ctypes.c_void_p
        for name, arguments, result in (
            ("create", [ctypes.c_char_p], pointer),
            ("free", [pointer], None),
            ("reset", [pointer], None),
            ("frame", [pointer, pointer, ctypes.POINTER(ctypes.c_float)], ctypes.c_int),
            ("state", [pointer, pointer], ctypes.c_int),
        ):
            function = getattr(lib, f"akiki_ten_vad_{name}")
            function.argtypes, function.restype = arguments, result

    def reset(self):
        super().reset()
        self.lib.akiki_ten_vad_reset(self.handle)

    def infer(self, window, state):
        if window.shape != (self.chunk_samples,):
            raise ValueError("TEN-VAD requires exactly 256 samples per frame")
        chunk = np.ascontiguousarray(window, dtype=np.float32)
        probability = ctypes.c_float()
        if self.lib.akiki_ten_vad_frame(self.handle, chunk.ctypes.data, ctypes.byref(probability)):
            raise RuntimeError("Akiki TEN-VAD engine could not score an audio frame")
        hidden = np.empty(self.state_shape, dtype=np.float32)
        if self.lib.akiki_ten_vad_state(self.handle, hidden.ctypes.data):
            raise RuntimeError("Akiki TEN-VAD engine could not return its recurrent state")
        return probability.value, hidden

    def close(self):
        super().close()
        self._finalizer()


class TenVadOnnxBackend(TenVadBackend):
    # The upstream streaming wrapper resets recurrent state every 30 seconds.
    _reset_frames = 1875

    def __init__(self, path: Path, lib, model_data: bytes):
        session = _create_onnx_session(_require_onnx_runtime(), model_data)
        self._input_names = tuple(value.name for value in session.get_inputs())
        if len(self._input_names) != 5 or len(session.get_outputs()) != 5:
            raise ValueError("Expected TEN-VAD features, probability, and four recurrent states")
        super().__init__(path, lib)
        self.session = session
        self._features = np.empty((1, 3, 41), dtype=np.float32)
        self._frames = 0

    @staticmethod
    def _bind(lib):
        TenVadBackend._bind(lib)
        pointer = ctypes.c_void_p
        lib.akiki_ten_vad_features.argtypes = [pointer, pointer, pointer]
        lib.akiki_ten_vad_features.restype = ctypes.c_int

    def reset(self):
        super().reset()
        self._frames = 0

    def infer(self, window, state):
        if window.shape != (self.chunk_samples,):
            raise ValueError("TEN-VAD requires exactly 256 samples per frame")
        chunk = np.ascontiguousarray(window, dtype=np.float32)
        if self.lib.akiki_ten_vad_features(
            self.handle, chunk.ctypes.data, self._features.ctypes.data
        ):
            raise RuntimeError("Akiki TEN-VAD frontend could not extract audio features")
        inputs = dict(zip(self._input_names, (self._features, *state[:, None, :]), strict=True))
        probability, *states = self.session.run(None, inputs)
        state = np.stack(states).reshape(self.state_shape)
        self._frames += 1
        if self._frames == self._reset_frames:
            state.fill(0)
            self._frames = 0
        return probability, state

    def close(self):
        super().close()
        self.session = None
