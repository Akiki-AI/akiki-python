import hashlib
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import akiki
from akiki import VAD, Engine, FFTPrecision, ModelPrecision
from akiki import _quantization as quant
from akiki.decision import ModelPrecision as MiniLMPrecision


@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.RUST])
def test_default_rust_precision_matches_explicit_int8(engine, tmp_path, require_rust):
    audio = np.random.default_rng(42).normal(0, 0.1, 256 * 9 + 71).astype(np.float32)
    with VAD(engine=engine, cache_dir=tmp_path) as default:
        assert default.engine is Engine.RUST
        assert (
            default.model_precision is ModelPrecision.INT8
            and default.fft_precision is FFTPrecision.FP32
        )
        expected = default.push(audio)
        path = default._backend.model_path
    with VAD(engine=engine, model_precision=ModelPrecision.INT8, cache_dir=tmp_path) as explicit:
        assert explicit._backend.model_path == path
        assert explicit.push(audio) == expected


@pytest.mark.parametrize("precision", list(ModelPrecision))
def test_precision_streaming_and_cache_reuse(precision, tmp_path, require_rust, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    audio = np.random.default_rng(42).normal(0, 0.1, 256 * 9 + 71).astype(np.float32)
    with VAD(model_precision=precision, cache_dir=tmp_path) as detector:
        expected = detector.push(audio)
        assert detector.engine is Engine.RUST
        assert detector.model_precision is precision and detector.fft_precision is FFTPrecision.FP32
        assert len(expected) == 9 and len(detector.pending) == 71
        assert all(np.isfinite(f.probability) and 0 <= f.probability <= 1 for f in expected)
        model_path = detector._backend.model_path
        detector.reset()
        actual = [
            f
            for start in range(0, len(audio), 173)
            for f in detector.push(audio[start : start + 173])
        ]
        assert actual == expected
    if precision is ModelPrecision.FP32:
        assert list(tmp_path.iterdir()) == []
        return
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    assert manifest["source"]["model_precision"] == precision.value
    before = model_path.stat().st_mtime_ns

    def conversion_must_not_repeat(*args):
        pytest.fail("A valid cache entry was unnecessarily converted again")

    monkeypatch.setattr(quant, "convert_model", conversion_must_not_repeat)
    with VAD(model_precision=precision, cache_dir=tmp_path) as detector:
        assert detector._backend.model_path == model_path
        assert detector.push(audio) == expected
    assert model_path.stat().st_mtime_ns == before


@pytest.mark.parametrize("damage", ["model", "manifest", "missing_manifest"])
def test_incomplete_or_corrupt_cache_is_regenerated(damage, tmp_path, require_rust):
    with VAD(model_precision=ModelPrecision.INT8, cache_dir=tmp_path) as detector:
        path = detector._backend.model_path
    original = path.read_bytes()
    if damage == "model":
        path.write_bytes(b"truncated")
    elif damage == "manifest":
        path.with_name("manifest.json").write_text("not json")
    else:
        path.with_name("manifest.json").unlink()
    with VAD(model_precision=ModelPrecision.INT8, cache_dir=tmp_path) as detector:
        assert len(detector.push(np.zeros(256))) == 1
    assert path.read_bytes() == original


def test_concurrent_first_use_produces_one_valid_variant(tmp_path, require_rust):
    def run(_):
        with VAD(model_precision=ModelPrecision.INT4, cache_dir=tmp_path) as detector:
            return detector._backend.model_path, detector.push(np.ones(256) * 0.01)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    assert all(result == results[0] for result in results)
    path = results[0][0]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    assert not list(tmp_path.rglob(".tmp-*"))


def test_converter_revision_invalidates_cache(tmp_path, require_rust, monkeypatch):
    with VAD(model_precision=ModelPrecision.BF16, cache_dir=tmp_path) as detector:
        first = detector._backend.model_path
    monkeypatch.setattr(quant, "CONVERTER_VERSION", quant.CONVERTER_VERSION + 1)
    with VAD(model_precision=ModelPrecision.BF16, cache_dir=tmp_path) as detector:
        second = detector._backend.model_path
    assert first != second
    assert first.read_bytes() == second.read_bytes()


def test_config_hash_invalidates_cache(tmp_path, require_rust):
    import shutil

    source = Path(akiki.__file__).parent / "models"
    shutil.copyfile(source / "akiki-vad.json", tmp_path / "custom.json")
    folder = tmp_path / "custom-native"
    shutil.copytree(source / "akiki-vad-native", folder)
    kwargs = dict(
        model_path=tmp_path / "custom.onnx",
        model_precision=ModelPrecision.INT8,
        cache_dir=tmp_path / "cache",
    )
    with VAD(**kwargs) as detector:
        first = detector._backend.model_path
    config = folder / "config.json"
    config.write_bytes(config.read_bytes() + b"\n")
    receipt = json.loads((folder / "export.json").read_text())
    receipt["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (folder / "export.json").write_text(json.dumps(receipt))
    with VAD(**kwargs) as detector:
        second = detector._backend.model_path
    assert first != second
    assert first.read_bytes() == second.read_bytes()


def test_cache_location_and_write_error_are_actionable(tmp_path, require_rust, monkeypatch):
    monkeypatch.setenv("AKIKI_CACHE_DIR", str(tmp_path / "cache"))
    with VAD(model_precision=ModelPrecision.FP16) as detector:
        assert detector._backend.model_path.is_relative_to(tmp_path / "cache")
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    with pytest.raises(RuntimeError, match="AKIKI_CACHE_DIR"):
        VAD(model_precision=ModelPrecision.INT8, cache_dir=blocked)


def test_precision_never_silently_falls_back_to_fp32(tmp_path):
    with pytest.raises(RuntimeError, match="requires a compatible Rust engine"):
        VAD(
            model_precision=ModelPrecision.INT8,
            library_path=tmp_path / "missing.so",
            cache_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "precision",
    [
        *(precision.value for precision in ModelPrecision),
        *MiniLMPrecision,
        "int3",
        FFTPrecision.FP32,
        Engine.AUTO,
        1,
        [],
    ],
)
def test_model_precision_requires_its_enum(precision):
    with pytest.raises(TypeError, match="model_precision must be a ModelPrecision member"):
        VAD(model_precision=precision)


@pytest.mark.parametrize(
    "precision",
    [
        "fp32",
        "int8",
        "fp16",
        "fp64",
        *ModelPrecision,
        Engine.AUTO,
        None,
        1,
        [],
    ],
)
def test_fft_precision_requires_its_enum(precision):
    with pytest.raises(TypeError, match="fft_precision must be an FFTPrecision member"):
        VAD(fft_precision=precision)


@pytest.mark.parametrize("engine", [Engine.ONNX, Engine.PYTORCH])
@pytest.mark.parametrize(
    "precision", [precision for precision in ModelPrecision if precision is not ModelPrecision.FP32]
)
def test_optional_engines_reject_reduced_precision(engine, precision):
    with pytest.raises(ValueError, match="precision"):
        VAD(engine=engine, model_precision=precision)


@pytest.mark.parametrize(
    "precision,limit",
    [
        (ModelPrecision.INT8, 127),
        (ModelPrecision.INT4, 7),
        (ModelPrecision.INT8_FP32_INPUT, 127),
    ],
)
def test_integer_conversion_scales_rounding_padding_and_sums(precision, limit):
    weights = np.array([[limit, -limit, limit / 2, -limit / 2], [0, 0, 0, 0]], dtype=np.float32)
    blob = quant._matrix(weights, np.zeros(2, np.float32), precision)
    assert struct.unpack_from("<III", blob) == (2, 4, 64)
    scales = np.frombuffer(blob, "<f4", count=2, offset=20)
    sums = np.frombuffer(blob, "<i4", count=2, offset=28)
    assert scales[0] == 1 and scales[1] > 0
    if precision is ModelPrecision.INT4:
        packed = np.frombuffer(blob, np.uint8, offset=36)
        assert packed[0] == 0x97 and packed[1] == 0xC4
        values = np.empty(128, np.int8)
        values[0::2] = ((packed & 15).astype(np.int16) ^ 8) - 8
        values[1::2] = ((packed >> 4).astype(np.int16) ^ 8) - 8
    else:
        values = np.frombuffer(blob, np.int8, offset=36)
    values = values.reshape(2, 64)
    assert list(values[0, :4]) == [limit, -limit, (limit + 1) // 2, -(limit + 1) // 2]
    assert not values[:, 4:].any() and not values[1].any()
    np.testing.assert_array_equal(sums, values.sum(axis=1))
    np.testing.assert_array_less(np.abs(values[:, :4] * scales[:, None] - weights), 0.50001)


@pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.BF16])
def test_float_conversion_preserves_representable_values(precision):
    weights = np.array([[0, -0.0, 1, -1, 0.5, 2]], dtype=np.float32)
    blob = quant._matrix(weights, np.array([0.125], np.float32), precision)
    assert struct.unpack_from("<III", blob) == (1, 6, 64)
    assert struct.unpack_from("<f", blob, 12)[0] == 0.125
    if precision is ModelPrecision.FP16:
        actual = np.frombuffer(blob, "<f2", offset=16).astype(np.float32)
    else:
        actual = (np.frombuffer(blob, "<u2", offset=16).astype("<u4") << 16).view("<f4")
    np.testing.assert_array_equal(actual[:6], weights[0])
    assert np.signbit(actual[1]) and not actual[6:].any()


def test_fp16_overflow_rejected():
    with pytest.raises(ValueError, match="finite FP16"):
        quant._matrix(np.array([[1e10]], np.float32), np.zeros(1, np.float32), ModelPrecision.FP16)


@pytest.mark.parametrize("precision", [None, "fp32", "int8", "bf16"])
def test_cli_uses_default_or_requested_precision(
    precision, tmp_path, require_rust, monkeypatch, capsys
):
    import soundfile as sf

    from akiki.__main__ import main

    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(256 * 3), 16000)
    arguments = [
        "akiki",
        "vad",
        str(audio),
        "--fft-precision",
        "fp32",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if precision is not None:
        arguments += ["--model-precision", precision]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(frames) == 3
    assert frames[-1]["end_seconds"] == 0.048
    variants = list((tmp_path / "cache").rglob("manifest.json"))
    if precision == "fp32":
        assert variants == []
    else:
        assert len(variants) == 1
        assert json.loads(variants[0].read_text())["source"]["model_precision"] == (
            precision or "int8"
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"dtype": "F16", "shape": [1], "data_offsets": [0, 4]},
        {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
        {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]},
        {"dtype": "F32", "shape": [1], "data_offsets": [0, 1000]},
    ],
)
def test_invalid_safetensors_metadata_rejected(metadata):
    header = json.dumps({"test": metadata}).encode()
    with pytest.raises(ValueError):
        quant._fp32_tensors(struct.pack("<Q", len(header)) + header + b"\0" * 4)
