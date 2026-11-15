"""Specialist training must reuse backend exports and preserve honest holdouts."""

import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from akiki._specialist_reports import evaluation_from_report
from akiki.decision import (
    DecisionModel,
    MiniLM,
    ModelPrecision,
    Specialist,
    TrainingAdvice,
)

backend = pytest.importorskip("akiki_training.training")
LocalTeacher = pytest.importorskip("akiki_training.teacher").LocalTeacher

EXAMPLES = {
    "w4 w5": "go",
    "w4 w6": "go",
    "w4 w7": "go",
    "w4 w8": "go",
    "w20 w21": "stop",
    "w20 w22": "stop",
    "w20 w23": "stop",
    "w20 w24": "stop",
}
VALIDATION = {
    "w4 w9": "go",
    "w4 w10": "go",
    "w4 w11": "go",
    "w20 w25": "stop",
    "w20 w26": "stop",
    "w20 w27": "stop",
}


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous)


@pytest.fixture
def configured(tmp_path, tiny_minilm):
    specialist = Specialist(
        intents={"go", "stop"}, examples=EXAMPLES.copy(), cache_dir=tmp_path / "cache"
    )
    config = backend.TrainingConfig(
        epochs=2, minimum_epochs=1, patience=2, batch_size=4, head_learning_rate=0.05
    )
    return specialist, tiny_minilm.plain, config


def test_training_saves_complete_adapter_reuses_cache_and_evaluates_after_restart(
    configured, runtime, monkeypatch, tmp_path
):
    specialist, base, config = configured
    trained = specialist.train(base, config=config, validation_examples=VALIDATION)
    assert not trained.cache_hit
    assert trained.labels == ("go", "stop")
    assert (trained.output_dir / "base_model").is_symlink()
    assert (trained.output_dir / "model_config.json").is_file()
    assert specialist.affine_layers.get(trained.base_model) is not None
    report = specialist.eval()
    assert report.examples == 6
    assert set(report.per_intent) == {"go", "stop"}
    for row in report.per_intent.values():
        assert row.examples == 3 and row.training_examples == 4
        assert 0 <= row.accuracy <= 1 and row.cross_entropy > 0
        assert (
            row.gradient_norm > 0 and row.affine_gradient_norm >= 0 and row.head_gradient_norm > 0
        )
        assert row.gradient_norm == pytest.approx(
            math.hypot(row.affine_gradient_norm, row.head_gradient_norm)
        )
        assert row.loss_slope is not None
        assert isinstance(row.advice, TrainingAdvice)
    assert report.cross_entropy == pytest.approx(
        sum(row.cross_entropy * row.examples for row in report.per_intent.values())
        / report.examples
    )
    assert "Affine gradient" in report._repr_html_()

    def must_not_retrain(**kwargs):
        pytest.fail("A valid saved specialist was unnecessarily retrained")

    from akiki import training

    with monkeypatch.context() as patch:
        patch.setattr(training, "train_intents", must_not_retrain)
        reused = specialist.train(base, config=config, validation_examples=VALIDATION)
    assert reused.cache_hit
    assert reused.output_dir == trained.output_dir
    other = Specialist(
        intents={"stop", "go"},
        examples=dict(reversed(list(EXAMPLES.items()))),
        cache_dir=specialist.cache_dir,
    )
    assert other.eval().cross_entropy == report.cross_entropy
    before = trained.adapter_path.stat().st_mtime_ns
    code = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'akiki_training', 'huggingface_hub'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from akiki.decision import Specialist
specialist = Specialist(intents={'go', 'stop'}, examples=json.loads(sys.argv[1]), cache_dir=sys.argv[2])
assert specialist.eval().examples == 6
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(EXAMPLES), str(specialist.cache_dir)],
        cwd=tmp_path,
        env=os.environ
        | {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert trained.adapter_path.stat().st_mtime_ns == before


def test_export_keeps_the_classification_head_and_matches_native_scores(
    configured, runtime, require_minilm_library
):
    specialist, base, config = configured
    trained = specialist.train(base, config=config, validation_examples=VALIDATION)
    with (
        MiniLM(base, trained.adapter_path, model_precision=ModelPrecision.FP32) as model,
        DecisionModel(base, model_precision=ModelPrecision.FP32) as decisions,
    ):
        decisions.Specialists.add(specialist)
        losses = {label: [] for label in trained.labels}
        correct = {label: [] for label in trained.labels}
        for text, label in VALIDATION.items():
            logits = model.encode(text).logits.astype(float)
            predicted = decisions.classify(text)
            np.testing.assert_allclose(predicted.logits, logits, atol=1e-6)
            assert predicted.label == trained.labels[predicted.index]
            shifted = logits - logits.max()
            log_probs = shifted - np.log(np.exp(shifted).sum())
            index = trained.labels.index(label)
            losses[label].append(-log_probs[index])
            correct[label].append(np.argmax(logits) == index)
    report = specialist.eval()
    for label in trained.labels:
        assert report.per_intent[label].cross_entropy == pytest.approx(
            np.mean(losses[label]), abs=2e-4
        )
        assert report.per_intent[label].accuracy == pytest.approx(np.mean(correct[label]))


def test_continuation_warm_starts_and_preserves_the_automatic_holdout(configured, runtime):
    specialist, base, config = configured
    first = specialist.train(base, config=config)
    initial = specialist.eval()
    old = json.loads((first.output_dir / "training.json").read_text())
    specialist.examples.update({"w4 w12": "go", "w20 w28": "stop"})
    with pytest.raises(ValueError, match="changed"):
        specialist.eval()
    continued = specialist.train(base, config=config, continue_training=True)
    new = json.loads((continued.output_dir / "training.json").read_text())
    assert (
        new["split_label_by_text_sha256"]["validation"]
        == old["split_label_by_text_sha256"]["validation"]
    )
    assert new["initial_validation"]["nll"] == pytest.approx(initial.cross_entropy, abs=1e-6)
    assert new["validation"]["nll"] <= new["initial_validation"]["nll"] + 1e-7
    assert new["training"]["examples"] == old["training"]["examples"] + 2
    assert first.output_dir != continued.output_dir and first.adapter_path.is_file()
    assert json.loads((continued.output_dir / "model_config.json").read_text())[
        "resumed_from"
    ] == str(first.output_dir)


def test_continuation_requires_the_same_explicit_holdout(configured, runtime):
    specialist, base, config = configured
    specialist.train(base, config=config, validation_examples=VALIDATION)
    with pytest.raises(ValueError, match="original validation_examples"):
        specialist.train(base, config=config, continue_training=True)
    changed = VALIDATION | {"w4 w13": "go"}
    with pytest.raises(ValueError, match="must stay unchanged"):
        specialist.train(base, config=config, validation_examples=changed, continue_training=True)


def test_teacher_proposals_are_reviewable_deduplicated_and_exclude_holdouts(configured, runtime):
    specialist, base, config = configured
    specialist.train(base, config=config, validation_examples=VALIDATION)

    class Teacher:
        def generate_examples(self, **kwargs):
            assert kwargs["intents"] == ["go", "stop"]
            return {" W4   W5 ": "go", "w4 w9": "go", "w4 w12": "go", "w20 w28": "stop"}

    proposed = specialist.suggest_examples(Teacher(), examples_per_intent=2)
    assert proposed == {"w4 w12": "go", "w20 w28": "stop"}
    assert specialist.examples == EXAMPLES
    assert specialist.eval().examples == 6
    restored = Specialist(
        intents=specialist.intents.copy(),
        examples=specialist.examples.copy(),
        cache_dir=specialist.cache_dir,
    )
    assert restored.suggest_examples(Teacher(), examples_per_intent=2) == proposed


def test_teacher_requires_an_existing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="existing local path"):
        LocalTeacher(tmp_path / "not-local").generate_examples(intents=["go"], examples={})


def test_local_teacher_forces_offline_loading_and_returns_proposals(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompt, **kwargs):
            return Batch(input_ids=np.array([[1, 2]]))

        def decode(self, value, **kwargs):
            return '["new go sample"]'

    class Model:
        config = SimpleNamespace(max_position_embeddings=1000)

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return np.array([[1, 2, 3]])

    def load_tokenizer(path, **options):
        calls.append(options)
        return Tokenizer()

    def load_model(path, **options):
        calls.append(options)
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    result = LocalTeacher(tmp_path, max_new_tokens=20).generate_examples(
        intents=["go"], examples={"existing": "go"}, examples_per_intent=1
    )
    assert result == {"new go sample": "go"}
    assert all(
        options == {"local_files_only": True, "trust_remote_code": False} for options in calls
    )
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_advice_distinguishes_improvement_plateau_and_insufficient_holdout():
    def metric(loss, accuracy=0.5):
        return {"examples": 5, "nll": loss, "accuracy": accuracy}

    training = metric(0.2) | {
        "gradient_norm": 0.1,
        "affine_gradient_norm": 0.08,
        "head_gradient_norm": 0.06,
    }
    report = {
        "labels": ["go"],
        "best_epoch": 2,
        "training": {"per_intent": [training]},
        "validation": metric(0.6) | {"per_intent": [metric(0.6)]},
        "history": [
            {"epoch": i, "validation": {"per_intent": [metric(loss)]}}
            for i, loss in enumerate([0.8, 0.7, 0.6], 1)
        ],
    }
    evaluation = evaluation_from_report(report, Path("saved"))
    assert evaluation.per_intent["go"].advice is TrainingAdvice.CONTINUE_TRAINING
    for epoch in report["history"]:
        epoch["validation"]["per_intent"][0]["nll"] = 0.6
    assert evaluation_from_report(report, Path("saved")).intents_needing_examples == {"go"}
    report["validation"]["per_intent"][0]["examples"] = 1
    assert (
        evaluation_from_report(report, Path("saved")).per_intent["go"].advice
        is TrainingAdvice.MORE_VALIDATION_DATA
    )
