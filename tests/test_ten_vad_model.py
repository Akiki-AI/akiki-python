"""Test TEN-VAD source selection and native cache handling offline."""

import hashlib
import sys
from types import SimpleNamespace

import pytest

from akiki.audio import _ten_vad_model


def test_pinned_snapshot_is_resolved_without_network(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return tmp_path

    sources = {
        _ten_vad_model._ONNX: b"ONNX model",
        _ten_vad_model._COEFFICIENTS: b"frontend coefficients",
    }
    for name, data in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    monkeypatch.setattr(
        _ten_vad_model,
        "_HUB_CHECKSUMS",
        {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
    )

    assert _ten_vad_model._read_sources(None) == tuple(sources.values())
    assert calls[0]["repo_id"] == "TEN-framework/ten-vad"
    assert calls[0]["revision"] == _ten_vad_model._REVISION
    assert calls[0]["local_files_only"] is True
    assert calls[0]["allow_patterns"] == list(sources)

    (tmp_path / _ten_vad_model._ONNX).write_bytes(b"changed model")
    with pytest.raises(ValueError, match="incorrect checksum"):
        _ten_vad_model._read_sources(None)


def test_local_onnx_reuses_and_repairs_native_cache(tmp_path, monkeypatch):
    model = tmp_path / "src/onnx_model/ten-vad.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"local weights")
    (model.parent.parent / "coeff.h").write_bytes(b"local coefficients")
    calls = []
    payload = _ten_vad_model._NATIVE_MAGIC + bytes(_ten_vad_model._NATIVE_SIZE - 8)

    def convert(weights, coefficients):
        calls.append((weights, coefficients))
        return payload

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(_ten_vad_model, "_convert", convert)

    cache = tmp_path / "cache"
    cached = _ten_vad_model.prepare_model(model, cache)
    assert cached.read_bytes() == payload
    assert _ten_vad_model.prepare_model(model, cache) == cached
    assert calls == [(b"local weights", b"local coefficients")]

    cached.write_bytes(payload[:-1] + b"\x01")
    assert _ten_vad_model.prepare_model(model, cache) == cached
    assert cached.read_bytes() == payload
    assert len(calls) == 2


def test_ten_vad_rejects_native_files_as_public_input(tmp_path):
    with pytest.raises(ValueError, match="must point to its .onnx weights"):
        _ten_vad_model.prepare_model(tmp_path / "weights.tnv", tmp_path / "cache")
