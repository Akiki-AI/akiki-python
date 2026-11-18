import numpy as np
import pytest

from akiki.audio.vad import VAD, Engine, FFTPrecision, ModelPrecision


def test_primary_engine_uses_the_akiki_name():
    assert Engine.AKIKI.value == "akiki"
    assert str(Engine.AKIKI) == "akiki"
    assert Engine("akiki") is Engine.AKIKI


@pytest.mark.parametrize(
    "available,expected_order",
    [
        (Engine.AKIKI, [Engine.AKIKI]),
        (Engine.ONNX, [Engine.AKIKI, Engine.ONNX]),
        (Engine.PYTORCH, [Engine.AKIKI, Engine.ONNX, Engine.PYTORCH]),
    ],
)
def test_auto_tries_akiki_before_compatibility_fallbacks(
    monkeypatch, tmp_path, available, expected_order
):
    from akiki import _engines

    attempts = []
    selected_backend = object()

    def backend_for(choice):
        def load(*args, **kwargs):
            attempts.append(choice)
            if choice is not available:
                raise _engines.EngineUnavailable(f"{choice.value} is unavailable")
            return selected_backend

        return load

    for choice, name in (
        (Engine.AKIKI, "AkikiBackend"),
        (Engine.ONNX, "OnnxBackend"),
        (Engine.PYTORCH, "PytorchBackend"),
    ):
        monkeypatch.setattr(_engines, name, backend_for(choice))

    backend, selected, errors = _engines.load_engine(tmp_path / "model.onnx", {}, Engine.AUTO, None)
    assert selected is available
    assert backend is selected_backend
    assert attempts == expected_order
    assert list(errors) == expected_order[:-1]


@pytest.fixture(params=list(Engine))
def engine(request):
    if request.param in (Engine.ONNX, Engine.PYTORCH):
        pytest.importorskip("onnxruntime" if request.param is Engine.ONNX else "torch")
    if request.param is Engine.AKIKI:
        request.getfixturevalue("require_akiki")
    return request.param


def test_streaming_matches_one_shot_and_reset(engine):
    audio = np.random.default_rng(1729).normal(0, 0.02, 256 * 9 + 71).astype(np.float32)
    detector = VAD(engine=engine)
    assert isinstance(detector.engine, Engine)
    assert not isinstance(detector.engine, str)
    assert isinstance(detector.model_precision, ModelPrecision)
    assert not isinstance(detector.model_precision, str)
    assert detector.fft_precision is FFTPrecision.FP32
    assert detector.engine is not Engine.AUTO
    if engine is not Engine.AUTO:
        assert detector.engine is engine
    expected = detector.push(audio)
    assert len(expected) == 9 and len(detector.pending) == 71
    detector.reset()
    actual = [f for i in range(0, len(audio), 173) for f in detector.push(audio[i : i + 173])]
    assert actual == expected
    assert actual[0].start_seconds == 0 and actual[-1].end_seconds == 0.144
    detector.reset()
    assert detector.push(audio) == expected


def test_invalid_audio_does_not_mutate_state(engine):
    detector = VAD(engine=engine)
    detector.push(np.zeros(37, np.float32))
    before = detector.pending.copy(), detector.state.copy(), detector.context.copy()
    for audio, rate in [
        (np.zeros(20), 8000),
        (np.zeros((20, 2)), 16000),
        (np.array([np.nan]), 16000),
    ]:
        with pytest.raises(ValueError):
            detector.push(audio, rate)
    for actual, expected in zip((detector.pending, detector.state, detector.context), before):
        np.testing.assert_array_equal(actual, expected)


def test_threshold_range():
    for threshold in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="Threshold"):
            VAD(threshold=threshold)


def test_default_is_selected_release_checkpoint():
    import hashlib
    import json
    from pathlib import Path

    import akiki

    model_dir = Path(akiki.__file__).parent / "models"
    metadata = json.loads((model_dir / "akiki-vad.json").read_text())
    assert akiki.__version__ == "0.2.2"
    assert VAD().model_id == "teacher-only623-v2-s1730"
    assert metadata["origin"]["seed"] == 1730
    assert metadata["release_selection"]["checkpoint_window"] == 31
    assert (
        hashlib.sha256((model_dir / "akiki-vad.onnx").read_bytes()).hexdigest()
        == "0fab56bc4ea6725ec881c35973e04813a4963dc5bbf4bf5ccfe8fd7cf468c6f5"
    )


def test_engines_agree(engine):
    pytest.importorskip("onnxruntime")
    audio = np.random.default_rng(42).normal(0, 0.05, 16000).astype(np.float32)
    with (
        VAD(engine=Engine.ONNX) as reference,
        VAD(engine=engine, model_precision=ModelPrecision.FP32) as detector,
    ):
        expected = reference.push(audio)
        actual = detector.push(audio)
        np.testing.assert_allclose(
            [f.probability for f in actual], [f.probability for f in expected], atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(detector.state, reference.state, atol=1e-5, rtol=0)
        assert [(f.start_seconds, f.end_seconds, f.speech) for f in actual] == [
            (f.start_seconds, f.end_seconds, f.speech) for f in expected
        ]


def test_closed_detector_rejects_use(engine):
    detector = VAD(engine=engine)
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.push([])
    with pytest.raises(RuntimeError, match="closed"):
        detector.reset()


def test_auto_falls_back_for_missing_or_incompatible_library(tmp_path):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "engine.so"
    for present in [False, True]:
        if present:
            path.write_bytes(b"wrong architecture or invalid shared library")
        with VAD(library_path=path) as detector:
            assert detector.engine is Engine.ONNX
            assert detector.model_precision is ModelPrecision.FP32
            assert Engine.AKIKI in detector.engine_errors
        with pytest.raises(RuntimeError, match="No compatible Akiki engine"):
            VAD(engine=Engine.AKIKI, library_path=path)


def test_auto_falls_back_to_pytorch(monkeypatch, tmp_path):
    import sys

    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with VAD(library_path=tmp_path / "missing.so") as detector:
        assert detector.engine is Engine.PYTORCH
        assert detector.model_precision is ModelPrecision.FP32
        assert set(detector.engine_errors) == {Engine.AKIKI, Engine.ONNX}
        assert len(detector.push(np.zeros(256))) == 1


def test_missing_all_engines_has_actionable_error(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match=r"akiki:.*onnx:.*pytorch:") as error:
        VAD(library_path=tmp_path / "missing.so")
    assert "pip install 'akiki[onnx]'" in str(error.value)
    assert "pip install 'akiki[pytorch]'" in str(error.value)


@pytest.mark.parametrize(
    "selected,unavailable", [(Engine.PYTORCH, "onnxruntime"), (Engine.ONNX, "torch")]
)
def test_explicit_optional_engine_works_without_other_runtime(selected, unavailable, monkeypatch):
    import sys

    pytest.importorskip("torch" if selected is Engine.PYTORCH else "onnxruntime")
    monkeypatch.setitem(sys.modules, unavailable, None)
    with VAD(engine=selected) as detector:
        assert detector.engine is selected
        assert detector.model_precision is ModelPrecision.FP32
        assert len(detector.push(np.zeros(256))) == 1


@pytest.mark.parametrize(
    "selected,module", [(Engine.PYTORCH, "torch"), (Engine.ONNX, "onnxruntime")]
)
def test_missing_explicit_optional_engine_names_its_extra(selected, module, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError) as error:
        VAD(engine=selected)
    assert f"pip install 'akiki[{selected}]'" in str(error.value)


@pytest.mark.parametrize("precision", [ModelPrecision.FP32, ModelPrecision.INT8])
def test_corrupt_native_model_is_not_hidden_by_fallback(tmp_path, precision):
    import shutil
    from pathlib import Path

    import akiki

    models = Path(akiki.__file__).parent / "models"
    shutil.copyfile(models / "akiki-vad.json", tmp_path / "akiki-vad.json")
    shutil.copytree(models / "akiki-vad-native", tmp_path / "akiki-vad-native")
    (tmp_path / "akiki-vad-native/model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        VAD(
            model_path=tmp_path / "akiki-vad.onnx",
            model_precision=precision,
            cache_dir=tmp_path / "cache",
        )


@pytest.mark.parametrize("engine", ["auto", "akiki", "onnx", "pytorch", "unknown", None, 1, []])
def test_non_enum_engine_is_rejected(engine):
    with pytest.raises(TypeError, match="engine must be an Engine member"):
        VAD(engine=engine)


def test_akiki_does_not_import_alternate_runtimes(require_akiki, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with VAD() as detector:
        assert detector.engine is Engine.AKIKI
        assert len(detector.push(np.zeros(256))) == 1
