"""DecisionModel must classify saved specialists with the existing Rust binding."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from conftest import write_safetensors

from akiki import decision, vad
from akiki._affine_cache import _base_identity, _digest
from akiki._specialist_training import (
    TrainedSpecialist,
    _base_key,
    _register,
    _root,
    _write_json,
)
from akiki.decision import DecisionModel, MiniLM, ModelPrecision, Specialist


def save_export(specialist, base, tiny, *, run="first", offset=0):
    """Write a small complete adapter in the persistent training-export format."""
    output = _root(specialist) / _base_key(base) / run
    output.mkdir(parents=True)
    (output / "base_model").symlink_to(base.path, target_is_directory=True)
    rng = np.random.default_rng(91)
    width = base.embedding_dim
    write_safetensors(
        output / "adapter.safetensors",
        {
            "interface.proj.weight": np.eye(width) + rng.uniform(-0.03, 0.03, (width, width)),
            "interface.proj.bias": rng.uniform(-0.02, 0.02, width),
            "head.weight": tiny.head["head.weight"],
            "head.bias": tiny.head["head.bias"] + offset * np.arange(len(tiny.labels)),
        },
    )
    _write_json(output / "model_config.json", {"labels": tiny.labels, "model_family": "minilm"})
    _write_json(
        output / "specialist.json",
        {
            "context": {
                "base_model": _base_identity(base),
                "examples_digest": _digest(sorted(specialist.examples.items())),
            },
            "labels": tiny.labels,
            "adapter_sha256": hashlib.sha256(
                (output / "adapter.safetensors").read_bytes()
            ).hexdigest(),
        },
    )
    artifact = TrainedSpecialist(base, output, tiny.labels)
    _register(specialist, artifact)
    return artifact


@pytest.fixture
def configured(tmp_path, tiny_minilm):
    model = DecisionModel(tiny_minilm.plain, threads=1, model_precision=ModelPrecision.FP32)
    specialist = Specialist(
        intents=set(tiny_minilm.labels),
        examples={f"w{index + 4}": label for index, label in enumerate(tiny_minilm.labels)},
        cache_dir=tmp_path / "specialists",
    )
    artifact = save_export(specialist, model.base_model, tiny_minilm)
    model.Specialists.add(specialist)
    yield model, specialist, artifact
    model.close()


def test_native_classification_matches_saved_affine_head_and_label_order(
    configured, require_minilm_library
):
    model, specialist, artifact = configured
    adapter_before = artifact.adapter_path.read_bytes()
    base_before = (model.base_model.path / "model.safetensors").read_bytes()
    with MiniLM(artifact.output_dir, threads=1, model_precision=ModelPrecision.FP32) as expected:
        for text in ("w4 w9", "w20 w25", ""):
            assert model.classify(text) == expected.classify(text)
            assert model.classify(text, specialist=specialist) == expected.classify(text)
    runtime = model._runtimes[specialist][1]
    assert runtime.threads == 1
    assert runtime.library_path == require_minilm_library
    assert sum(model.classify("w4").probabilities) == pytest.approx(1)
    assert artifact.adapter_path.read_bytes() == adapter_before
    assert (model.base_model.path / "model.safetensors").read_bytes() == base_before


def test_native_options_and_handles_are_reused(configured, require_minilm_library, monkeypatch):
    model, specialist, artifact = configured
    with DecisionModel(
        model.base_model,
        threads=2,
        model_precision=ModelPrecision.INT8,
        library_path=require_minilm_library,
    ) as quantized:
        quantized.Specialists.add(specialist)
        first = quantized.classify("w4 w5")
        runtime = quantized._runtimes[specialist][1]
        assert runtime.threads == 2 and runtime.model_precision is ModelPrecision.INT8

        def must_not_reload(*args, **kwargs):
            pytest.fail("Repeated classification reloaded the native model")

        monkeypatch.setattr(decision, "MiniLM", must_not_reload)
        assert quantized.classify("w4 w5") == first
        with ThreadPoolExecutor(max_workers=3) as pool:
            assert list(pool.map(quantized.classify, ["w4 w5"] * 6)) == [first] * 6
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4 w5")


def test_multiple_specialists_keep_results_separate(
    configured, tiny_minilm, require_minilm_library
):
    model, first, first_artifact = configured
    second = Specialist(intents=first.intents.copy(), examples={"w12": tiny_minilm.labels[0]})
    second.cache_dir = first.cache_dir
    second_artifact = save_export(second, model.base_model, tiny_minilm, offset=0.5)
    model.Specialists.add(second)
    with pytest.raises(ValueError, match="pass specialist explicitly"):
        model.classify("w4 w5")
    results = model.decide("w4 w5")
    assert set(results) == {first, second}
    assert results[first].logits != results[second].logits
    for specialist, artifact in ((first, first_artifact), (second, second_artifact)):
        with MiniLM(artifact.output_dir, threads=1, model_precision=ModelPrecision.FP32) as direct:
            assert results[specialist] == direct.classify("w4 w5")
        assert results[specialist] == model.classify("w4 w5", specialist=specialist)


def test_new_saved_training_run_replaces_loaded_handle(
    configured, tiny_minilm, require_minilm_library
):
    model, specialist, artifact = configured
    first = model.classify("w4 w5")
    old_runtime = model._runtimes[specialist][1]
    continued = save_export(specialist, model.base_model, tiny_minilm, run="continued", offset=1)
    result = model.classify("w4 w5")
    assert result.logits != first.logits
    with MiniLM(continued.output_dir, threads=1, model_precision=ModelPrecision.FP32) as direct:
        assert result == direct.classify("w4 w5")
    with pytest.raises(RuntimeError, match="closed"):
        old_runtime.classify("w4 w5")
    assert artifact.adapter_path.is_file()


def test_configuration_changes_release_native_handles(configured, tmp_path, require_minilm_library):
    model, specialist, artifact = configured
    expected = model.classify("w4")
    runtime = model._runtimes[specialist][1]
    with pytest.raises(FileNotFoundError):
        model.base_model = tmp_path / "missing"
    assert model.classify("w4") == expected
    assert model._runtimes[specialist][1] is runtime

    model.Specialists.remove(specialist)
    with pytest.raises(ValueError, match="Add a trained Specialist"):
        model.classify("w4")
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4")

    model.Specialists.add(specialist)
    model.classify("w4")
    runtime = model._runtimes[specialist][1]
    model.Specialists = set()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4")

    model.Specialists.add(specialist)
    model.classify("w4")
    runtime = model._runtimes[specialist][1]
    other = tmp_path / "other-base"
    shutil.copytree(model.base_model.path, other)
    model.base_model = other
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4")
    with pytest.raises(ValueError, match="no saved training result"):
        model.classify("w4")
    model.base_model = None
    with pytest.raises(ValueError, match="base_model"):
        model.classify("w4")


def test_changed_data_and_damaged_adapters_never_use_old_weights(
    configured, require_minilm_library
):
    model, specialist, artifact = configured
    model.classify("w4")
    runtime = model._runtimes[specialist][1]
    specialist.examples["w30"] = next(iter(specialist.intents))
    with pytest.raises(ValueError, match="no saved training result"):
        model.classify("w4")
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4")
    specialist.examples.pop("w30")
    model.classify("w4")
    runtime = model._runtimes[specialist][1]
    artifact.adapter_path.write_bytes(artifact.adapter_path.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.classify("w4")
    with pytest.raises(RuntimeError, match="closed"):
        runtime.classify("w4")


def test_unsupported_family_and_inconsistent_exports_fail_before_native_loading(
    configured, monkeypatch
):
    model, specialist, artifact = configured

    def must_not_load(*args, **kwargs):
        pytest.fail("An invalid export reached the native loader")

    monkeypatch.setattr(decision, "MiniLM", must_not_load)
    config = artifact.output_dir / "model_config.json"
    _write_json(config, {"labels": artifact.labels, "model_family": "qwen3.5"})
    with pytest.raises(NotImplementedError, match="qwen3.5.*Python binding"):
        model.classify("w4")
    _write_json(config, {"labels": list(reversed(artifact.labels)), "model_family": "minilm"})
    with pytest.raises(ValueError, match="label order"):
        model.classify("w4")
    _write_json(config, {"labels": artifact.labels, "model_family": "minilm"})
    link = artifact.output_dir / "base_model"
    link.unlink()
    link.symlink_to(artifact.output_dir)
    with pytest.raises(ValueError, match="base_model does not match"):
        model.classify("w4")


def test_inference_errors_and_lifecycle(configured, monkeypatch):
    model, specialist, artifact = configured

    def must_not_load(*args, **kwargs):
        pytest.fail("Invalid inference started the native runtime")

    monkeypatch.setattr(decision, "MiniLM", must_not_load)
    with pytest.raises(TypeError, match="text must be a string"):
        model.classify(["w4"])
    with pytest.raises(TypeError, match="specialist must be"):
        model.classify("w4", specialist="wrong")
    with pytest.raises(ValueError, match="Add this specialist"):
        model.classify("w4", specialist=Specialist())
    model.Specialists.add("wrong")
    with pytest.raises(TypeError, match="only Specialist"):
        model.decide("w4")
    model.Specialists.remove("wrong")
    model.Specialists = {Specialist(intents=specialist.intents)}
    with pytest.raises(ValueError, match="no saved training result"):
        model.classify("w4")
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.classify("w4")
    with pytest.raises(RuntimeError, match="closed"):
        model.__enter__()


def test_missing_native_library_has_no_runtime_fallback(configured, tmp_path):
    model, specialist, artifact = configured
    with DecisionModel(model.base_model, library_path=tmp_path / "missing.so") as unavailable:
        unavailable.Specialists.add(specialist)
        with pytest.raises(RuntimeError, match="AKIKI_MINILM_LIBRARY"):
            unavailable.classify("w4")
        assert not unavailable._runtimes


@pytest.mark.parametrize("precision", ["fp32", None, vad.ModelPrecision.FP32])
def test_precision_requires_the_decision_enum(precision):
    with pytest.raises(TypeError, match="akiki.decision.ModelPrecision"):
        DecisionModel(model_precision=precision)


@pytest.mark.parametrize("threads", [0, 257, 1.5, True])
def test_thread_options_are_validated_without_loading(threads):
    with pytest.raises(ValueError, match="threads"):
        DecisionModel(threads=threads)


def test_saved_inference_after_restart_needs_no_training_runtime_or_network(
    configured, tmp_path, require_minilm_library
):
    model, specialist, artifact = configured
    expected = model.classify("w4 w5")
    code = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'akiki_training', 'huggingface_hub', 'safetensors'}:
            raise AssertionError(fullname)
def block_network(event, args):
    if event == 'socket.connect':
        raise AssertionError('Inference attempted network access')
sys.meta_path.insert(0, Block())
sys.addaudithook(block_network)
from akiki.decision import DecisionModel, ModelPrecision, Specialist
specialist = Specialist(intents=set(json.loads(sys.argv[1])), examples=json.loads(sys.argv[2]), cache_dir=sys.argv[3])
with DecisionModel(sys.argv[4], threads=1, model_precision=ModelPrecision.FP32, library_path=sys.argv[5]) as model:
    model.Specialists.add(specialist)
    result = model.classify('w4 w5')
    print(json.dumps({'label': result.label, 'logits': result.logits}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(sorted(specialist.intents)),
            json.dumps(specialist.examples),
            str(specialist.cache_dir),
            str(model.base_model.path),
            str(require_minilm_library),
        ],
        cwd=tmp_path,
        env=os.environ
        | {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
    assert restored["label"] == expected.label
    assert restored["logits"] == pytest.approx(expected.logits, abs=1e-7)


def test_tokenization_uses_the_selected_specialist_and_reuses_its_handle(configured, monkeypatch):
    model, specialist, artifact = configured
    with MiniLM(artifact.output_dir, threads=1, model_precision=ModelPrecision.FP32) as direct:
        for special_tokens in (True, False):
            expected = direct.tokenize("w4 w5", special_tokens=special_tokens)
            assert model.tokenize("w4 w5", special_tokens=special_tokens) == expected
            assert (
                model.tokenize("w4 w5", special_tokens=special_tokens, specialist=specialist)
                == expected
            )
        long_text = "w4 " * 600
        assert model.tokenize(long_text) == direct.tokenize(long_text)
        assert len(model.tokenize(long_text)) > direct.max_tokens
    runtime = model._runtimes[specialist][1]

    def must_not_reload(*args, **kwargs):
        pytest.fail("Tokenization reloaded the native handle")

    monkeypatch.setattr(decision, "MiniLM", must_not_reload)
    assert model.classify("w4 w5").label in specialist.intents
    model.tokenize("w4 w5")
    assert model._runtimes[specialist][1] is runtime
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.tokenize("w4")


def test_tokenization_uses_the_same_specialist_selection_rules(configured):
    model, specialist, artifact = configured
    other = Specialist(intents=specialist.intents.copy())
    with pytest.raises(ValueError, match="Add this specialist"):
        model.tokenize("w4", specialist=other)
    model.Specialists.add(other)
    with pytest.raises(ValueError, match="pass specialist explicitly"):
        model.tokenize("w4")
    assert model.tokenize("w4", specialist=specialist)
