"""Object detection in RGB images with the local Akiki engine.

Use ObjectDetector with the cached Akiki-AI/akiki-cam-seen snapshot.
Each Detection gives an object family, a score, and a box in image pixels.
"""

import ctypes
import math
import os
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._engines import library_candidates

__all__ = ["Detection", "ObjectDetector"]

_MODEL_ID = "Akiki-AI/akiki-cam-seen"
_FILES = (
    "seen-r4c-56508271-model.json",
    "seen-r4c-56508271-weights.json",
    "seen-r4c-56508271-weights.bin",
)
_MAX_DETECTIONS = 100


@dataclass(frozen=True)
class Detection:
    """One object family and its bounding box.

    The box gives the left, top, right, and bottom edges in source image pixels.
    """

    family: str
    word: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float


class _NativeDetection(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("x0", ctypes.c_float),
        ("y0", ctypes.c_float),
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
    ]


def _cached_snapshot(cache_dir: str | Path | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError as error:
        raise RuntimeError(
            "Install akiki[huggingface] to find the cached Seen model, "
            "or pass model_path for a local snapshot"
        ) from error
    try:
        return Path(
            snapshot_download(
                repo_id=_MODEL_ID,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise FileNotFoundError(
            f"The Hugging Face cache has no snapshot for {_MODEL_ID}. "
            "Pass model_path for an existing local snapshot"
        ) from error


def _load_library(library_path: str | Path | None) -> tuple[ctypes.CDLL, Path]:
    candidate = next(library_candidates(library_path, "akiki_seen_engine", "AKIKI_SEEN_LIBRARY"))
    path = candidate.resolve()
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    bindings = {
        "load": ([ctypes.c_char_p, ctypes.POINTER(size), size], pointer),
        "free": ([pointer], None),
        "family_count": ([pointer], size),
        "family_id": ([pointer, size], ctypes.c_char_p),
        "family_word": ([pointer, size], ctypes.c_char_p),
        "last_error": ([], ctypes.c_char_p),
        "detect": (
            [
                pointer,
                pointer,
                ctypes.c_uint32,
                ctypes.c_uint32,
                size,
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_int32,
                ctypes.POINTER(_NativeDetection),
                size,
            ],
            ctypes.c_int64,
        ),
    }
    try:
        library = ctypes.CDLL(str(path))
        for name, (args, result) in bindings.items():
            function = getattr(library, f"akiki_seen_{name}")
            function.argtypes = args
            function.restype = result
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            f"No compatible Akiki Seen library at {path}: {error}. "
            "Set AKIKI_SEEN_LIBRARY or pass library_path for a local library"
        ) from error
    return library, path


def _error_text(library: ctypes.CDLL) -> str:
    return (library.akiki_seen_last_error() or b"unknown engine error").decode("utf-8", "replace")


class ObjectDetector:
    """Find the object families supported by Akiki Cam Seen.

    Pass a local Hugging Face snapshot folder as model_path. If you omit it,
    the detector looks in the configured Hugging Face cache. Use cache_dir to
    select a different cache folder. The cache lookup is local only.

    Pass library_path or set AKIKI_SEEN_LIBRARY to select a local Akiki engine
    library. Pass cpus to pin each engine worker to one CPU. If you omit cpus,
    the engine selects the CPU placement.

    Use a with block or call close() to release the engine.
    """

    model_id = _MODEL_ID

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        library_path: str | Path | None = None,
        cpus: Sequence[int] | None = None,
    ) -> None:
        cpu_ids = tuple(cpus) if cpus is not None else ()
        max_cpu = 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1
        if any(type(cpu) is not int or not 0 <= cpu <= max_cpu for cpu in cpu_ids):
            raise ValueError("cpus must contain nonnegative integer CPU IDs")
        if len(set(cpu_ids)) != len(cpu_ids):
            raise ValueError("cpus must contain distinct CPU IDs")

        root = Path(model_path) if model_path is not None else _cached_snapshot(cache_dir)
        self.model_path = root.expanduser().resolve()
        weights_dir = self.model_path / "weights"
        missing = [name for name in _FILES if not (weights_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing Seen model files in {weights_dir}: {', '.join(missing)}"
            )

        library, self.library_path = _load_library(library_path)
        native_cpus = (ctypes.c_size_t * len(cpu_ids))(*cpu_ids) if cpu_ids else None
        handle = library.akiki_seen_load(os.fsencode(weights_dir), native_cpus, len(cpu_ids))
        if not handle:
            raise ValueError(f"Akiki engine cannot load {self.model_path}: {_error_text(library)}")

        self._library = library
        self._handle = handle
        self._lock = threading.Lock()
        self._finalizer = weakref.finalize(self, library.akiki_seen_free, handle)
        try:
            self._families = tuple(
                (
                    library.akiki_seen_family_id(handle, index).decode("utf-8"),
                    library.akiki_seen_family_word(handle, index).decode("utf-8"),
                )
                for index in range(library.akiki_seen_family_count(handle))
            )
        except Exception:
            self.close()
            raise

    def detect(
        self, image: np.ndarray, *, threshold: float | None = None, policy: bool = True
    ) -> list[Detection]:
        """Find objects in an image with shape (height, width, 3) and dtype uint8.

        Use RGB channel order. Omit threshold to use the model's score threshold,
        or give a value greater than zero and at most one. Set policy to False
        to skip the model's per-family score limits. Results are in score order.
        """
        with self._lock:
            self._check_open()
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected an RGB uint8 image with shape (height, width, 3)")
            height, width = image.shape[:2]
            if not (0 < height < 2**32 and 0 < width < 2**32):
                raise ValueError("Image height and width must be positive 32-bit dimensions")
            confidence = 0.0 if threshold is None else float(threshold)
            if threshold is not None and not (math.isfinite(confidence) and 0 < confidence <= 1):
                raise ValueError("threshold must be greater than zero and at most one")
            if not isinstance(policy, bool):
                raise TypeError("policy must be a bool")

            pixels = np.ascontiguousarray(image)
            results = (_NativeDetection * _MAX_DETECTIONS)()
            count = self._library.akiki_seen_detect(
                self._handle,
                pixels.ctypes.data,
                width,
                height,
                0,
                0,
                confidence,
                int(policy),
                results,
                len(results),
            )
            if count < 0:
                raise RuntimeError(f"Seen detection failed: {_error_text(self._library)}")
            return [
                Detection(
                    *self._families[item.family], item.score, item.x0, item.y0, item.x1, item.y1
                )
                for item in results[:count]
            ]

    def _check_open(self) -> None:
        if not self._finalizer.alive:
            raise RuntimeError("The object detector is closed")

    def close(self) -> None:
        """Release the engine. You can call this method more than once."""
        with self._lock:
            self._finalizer()

    def __enter__(self):
        """Return this open detector for use in a with block."""
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        """Release the engine on block exit."""
        self.close()
