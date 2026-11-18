"""Check the cached Seen model path and the native detector interface."""

import ctypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from akiki.vision import object_detection

FILES = (
    "seen-r4c-56508271-model.json",
    "seen-r4c-56508271-weights.json",
    "seen-r4c-56508271-weights.bin",
)


@pytest.fixture
def snapshot(tmp_path):
    folder = tmp_path / "snapshot"
    weights = folder / "weights"
    weights.mkdir(parents=True)
    for name in FILES:
        (weights / name).touch()
    return folder


@pytest.fixture
def cached_hub(monkeypatch):
    class CacheMiss(Exception):
        pass

    hub = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub.utils", SimpleNamespace(LocalEntryNotFoundError=CacheMiss)
    )
    return hub, CacheMiss


@pytest.fixture
def native(monkeypatch, tmp_path):
    library_path = tmp_path / "libseen"
    calls = SimpleNamespace(load=[], free=[], detect=[])

    def load(path, cpus, count):
        calls.load.append((Path(os.fsdecode(path)), tuple(cpus[i] for i in range(count))))
        return 123

    def detect(handle, pixels, width, height, stride, format_id, threshold, policy, out, size):
        data = (ctypes.c_uint8 * (width * height * 3)).from_address(pixels)
        calls.detect.append(
            (handle, width, height, stride, format_id, threshold, policy, bytes(data))
        )
        assert size >= 1
        out[0].family = 0
        out[0].score = 0.75
        out[0].x0, out[0].y0, out[0].x1, out[0].y1 = 1.0, 2.0, 3.0, 4.0
        return 1

    library = SimpleNamespace(
        akiki_seen_load=load,
        akiki_seen_free=calls.free.append,
        akiki_seen_family_count=lambda handle: 1,
        akiki_seen_family_id=lambda handle, index: b"teddy-bear",
        akiki_seen_family_word=lambda handle, index: b"teddy bear",
        akiki_seen_detect=detect,
    )
    monkeypatch.setattr(object_detection, "_load_library", lambda path: (library, library_path))
    return calls


def test_cached_seen_model_uses_local_files_only(snapshot, cached_hub, native, tmp_path):
    hub, _ = cached_hub
    cache_dir = tmp_path / "hub"
    requests = []

    def lookup(**kwargs):
        requests.append(kwargs)
        return str(snapshot)

    hub.snapshot_download = lookup
    with object_detection.ObjectDetector(cache_dir=cache_dir) as detector:
        assert detector.model_id == "Akiki-AI/akiki-cam-seen"
        assert detector.model_path == snapshot
    assert requests == [
        {
            "repo_id": "Akiki-AI/akiki-cam-seen",
            "cache_dir": cache_dir,
            "local_files_only": True,
        }
    ]
    assert native.load == [(snapshot / "weights", ())]
    assert native.free == [123]


def test_cache_miss_requests_existing_snapshot(cached_hub):
    hub, cache_miss = cached_hub

    def lookup(**kwargs):
        raise cache_miss

    hub.snapshot_download = lookup
    with pytest.raises(FileNotFoundError, match="model_path for an existing local snapshot"):
        object_detection.ObjectDetector()


def test_partial_snapshot_is_rejected_before_loading_an_engine(snapshot, native):
    (snapshot / "weights" / FILES[2]).unlink()
    with pytest.raises(FileNotFoundError, match=FILES[2]):
        object_detection.ObjectDetector(snapshot)
    assert native.load == []


def test_detection_copies_noncontiguous_rgb_and_frees_handle_once(snapshot, native):
    image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)[:, ::2]
    with object_detection.ObjectDetector(snapshot, cpus=(0, 1)) as detector:
        detection = detector.detect(image, threshold=0.5, policy=False)
    assert detection == [object_detection.Detection("teddy-bear", "teddy bear", 0.75, 1, 2, 3, 4)]
    assert native.load == [(snapshot / "weights", (0, 1))]
    assert native.detect == [(123, 3, 5, 0, 0, 0.5, 0, image.tobytes())]
    detector.close()
    assert native.free == [123]
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(image)


def test_invalid_rgb_and_threshold_are_rejected_before_ffi(snapshot, native):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    with object_detection.ObjectDetector(snapshot) as detector:
        with pytest.raises(ValueError, match="RGB uint8"):
            detector.detect(image.astype(np.float32))
        with pytest.raises(ValueError, match="positive 32-bit"):
            detector.detect(image[:0])
        with pytest.raises(ValueError, match="threshold"):
            detector.detect(image, threshold=float("nan"))
    assert native.detect == []
