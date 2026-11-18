"""Local binding for the Akiki TEN-VAD engine."""

import ctypes
import weakref
from pathlib import Path

import numpy as np

from .._engines import Backend, EngineUnavailable, library_candidates


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
