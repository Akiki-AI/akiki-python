"""Thin specialist orchestration around the optional akiki-training package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from ._affine_cache import AffineLayer, _AffineCache, _base_identity, _digest
from ._base_models import BaseModel, _resolve_base_model
from ._quantization import _fp32_tensors
from ._specialist_reports import SpecialistEvaluation, evaluation_from_report


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".tmp-", mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid specialist metadata: {path}")
    return value


def _text_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _root(specialist) -> Path:
    return _AffineCache(specialist)._namespace()[0] / "trained"


def _base_key(base: BaseModel) -> str:
    return _digest(_base_identity(base))


@dataclass(frozen=True)
class TrainedSpecialist:
    """A saved input-affine/head adapter and its training report.

    ``base_model`` identifies the local frozen checkpoint.
    ``output_dir`` contains the adapter, labels, reports, and a ``base_model``
    link. That local checkpoint must remain available for inference or training.
    ``labels`` gives the sorted intent order used by the classification head.
    ``cache_hit`` is true when this result was loaded from a matching saved export.

    This object describes a saved result. It does not hold a loaded inference
    model. Pass its export directory to ``MiniLM`` for direct inference, or use
    the corresponding ``Specialist`` with ``DecisionModel``.
    """

    base_model: BaseModel
    output_dir: Path
    labels: tuple[str, ...]
    cache_hit: bool = False

    @property
    def adapter_path(self) -> Path:
        """Return the path to the saved affine and head in ``adapter.safetensors``."""
        return self.output_dir / "adapter.safetensors"

    def eval(self) -> SpecialistEvaluation:
        """Read validation scores and diagnostics from this result's saved report.

        This loads no model and runs no new predictions. It needs no optional
        training runtime. Missing or incompatible reports raise an error.
        """
        return evaluation_from_report(
            _read_json(self.output_dir / "training.json"), self.output_dir
        )


def _load_artifact(path: Path, *, cache_hit=False) -> TrainedSpecialist:
    metadata = _read_json(path / "specialist.json")
    base = metadata["context"]["base_model"]
    model = BaseModel(Path(base["path"]), base["vocab_size"], base["embedding_dim"])
    expected = metadata["adapter_sha256"]
    if hashlib.sha256((path / "adapter.safetensors").read_bytes()).hexdigest() != expected:
        raise ValueError(f"Trained specialist adapter checksum mismatch: {path}")
    return TrainedSpecialist(model, path, tuple(metadata["labels"]), cache_hit)


def _latest(specialist, base=None) -> TrainedSpecialist:
    if base is None and specialist._last_trained is not None:
        return specialist._last_trained
    root = _root(specialist)
    pointers = (
        list(root.glob("*/latest.json"))
        if base is None
        else [root / _base_key(base) / "latest.json"]
    )
    pointers = [path for path in pointers if path.is_file()]
    if not pointers:
        raise ValueError(
            "This specialist has no saved training result for this data; call train() first"
        )
    if len(pointers) != 1:
        raise ValueError("Several base models have saved results; pass base_model explicitly")
    pointer = _read_json(pointers[0])
    return _load_artifact(pointers[0].parent / pointer["run"], cache_hit=True)


def _register(specialist, artifact: TrainedSpecialist) -> None:
    tensors = _fp32_tensors(artifact.adapter_path.read_bytes())
    specialist.affine_layers[artifact.base_model] = AffineLayer(
        tensors["interface.proj.weight"], tensors["interface.proj.bias"]
    )
    _write_json(
        _root(specialist) / _base_key(artifact.base_model) / "latest.json",
        {"run": artifact.output_dir.name},
    )
    specialist._last_trained = artifact


def train_specialist(
    specialist,
    base_model,
    *,
    config=None,
    validation_examples=None,
    test_examples=None,
    continue_training=False,
    resume_from=None,
    on_epoch=None,
) -> TrainedSpecialist:
    import akiki_training

    from . import training

    base = _resolve_base_model(base_model)
    if config is None:
        config = training.TrainingConfig()
    if not isinstance(config, training.TrainingConfig):
        raise TypeError("config must be an akiki.training.TrainingConfig")
    if not isinstance(continue_training, bool):
        raise TypeError("continue_training must be a bool")
    intents = sorted(specialist.intents)
    if len(intents) < 2 or any(
        not isinstance(intent, str) or not intent.strip() for intent in intents
    ):
        raise ValueError("A specialist needs at least two nonempty intent names")
    examples = dict(specialist.examples)
    previous = None
    validation_source = "explicit" if validation_examples is not None else "automatic"
    if continue_training or resume_from is not None:
        previous = (
            (
                resume_from
                if isinstance(resume_from, TrainedSpecialist)
                else _load_artifact(Path(resume_from).expanduser().resolve())
            )
            if resume_from is not None
            else _latest(specialist, base if specialist._last_trained is None else None)
        )
        if previous.base_model != base:
            raise ValueError("Cannot continue an adapter with another base model")
        if previous.labels != tuple(intents):
            raise ValueError("Cannot continue an adapter after changing the intent set")
        old_metadata = _read_json(previous.output_dir / "specialist.json")
        old_context = old_metadata["context"]
        old_report = _read_json(previous.output_dir / "training.json")
        if validation_examples is None:
            if old_metadata["validation_source"] == "explicit":
                raise ValueError(
                    "Pass the original validation_examples when continuing this specialist"
                )
            hashes = set(old_report["split_text_sha256"]["validation"])
            validation_examples = {
                text: label for text, label in examples.items() if _text_hash(text) in hashes
            }
            if {_text_hash(text) for text in validation_examples} != hashes:
                raise ValueError(
                    "Original validation examples changed; supply the original holdout explicitly"
                )
            for text in validation_examples:
                examples.pop(text)
        expected_validation = old_report["split_label_by_text_sha256"]["validation"]
        if {
            _text_hash(text): label for text, label in (validation_examples or {}).items()
        } != expected_validation:
            raise ValueError(
                "Validation examples and labels must stay unchanged during continuation"
            )
        if old_context["test_digest"] != _digest(sorted((test_examples or {}).items())):
            raise ValueError("Pass the same test_examples during continuation")
        examples = {
            text: label
            for text, label in examples.items()
            if _text_hash(text) not in expected_validation
        }
    serialized_config = {
        key: value.value if isinstance(value, Enum) else value
        for key, value in asdict(config).items()
    }
    context = {
        "version": 1,
        "backend_version": akiki_training.__version__,
        "base_model": _base_identity(base),
        "intents": intents,
        "examples_digest": _digest(sorted(specialist.examples.items())),
        "validation_digest": _digest(sorted((validation_examples or {}).items())),
        "test_digest": _digest(sorted((test_examples or {}).items())),
        "config": serialized_config,
        "resume_adapter_sha256": hashlib.sha256(previous.adapter_path.read_bytes()).hexdigest()
        if previous
        else None,
    }
    output = _root(specialist) / _base_key(base) / _digest(context)
    if output.exists():
        artifact = _load_artifact(output, cache_hit=True)
        if _read_json(output / "specialist.json")["context"] != context:
            raise ValueError(f"Saved specialist context does not match its cache key: {output}")
        _register(specialist, artifact)
        return artifact
    result = training.train_intents(
        intents=intents,
        examples=examples,
        base_model=base.path,
        output_dir=output,
        validation_examples=validation_examples,
        test_examples=test_examples,
        config=config,
        on_epoch=on_epoch,
        resume_from=previous.output_dir if previous else None,
        copy_base_model=False,
    )
    if Path(result.output_dir).resolve() != output.resolve():
        raise ValueError("Trainer returned an unexpected export directory")
    metadata = {
        "context": context,
        "labels": list(result.labels),
        "validation_source": validation_source,
        "adapter_sha256": hashlib.sha256((output / "adapter.safetensors").read_bytes()).hexdigest(),
    }
    _write_json(output / "specialist.json", metadata)
    artifact = _load_artifact(output)
    if context["examples_digest"] != _digest(sorted(specialist.examples.items())) or tuple(
        intents
    ) != tuple(sorted(specialist.intents)):
        raise ValueError(
            f"Specialist data changed while training; the completed result is preserved at {output}"
        )
    _register(specialist, artifact)
    return artifact


def _current_artifact(specialist, base=None) -> TrainedSpecialist:
    """Find a complete export for the current data without importing training."""
    artifact = _latest(specialist, base)
    metadata = _read_json(artifact.output_dir / "specialist.json")
    if base is not None and artifact.base_model != base:
        raise ValueError("Saved specialist was trained with another base model")
    if (
        metadata["context"]["examples_digest"] != _digest(sorted(specialist.examples.items()))
        or tuple(sorted(specialist.intents)) != artifact.labels
    ):
        raise ValueError(
            "Specialist intents or examples changed; train the current data before using it"
        )
    return artifact


def evaluate_specialist(specialist, base_model=None) -> SpecialistEvaluation:
    base = _resolve_base_model(base_model) if base_model is not None else None
    return _current_artifact(specialist, base).eval()


def suggest_examples(
    specialist, teacher, *, examples_per_intent=5, intents=None, excluded_examples=None
) -> dict[str, str]:
    from akiki_training.teacher import LocalTeacher, validate_suggestions

    if type(examples_per_intent) is not int or examples_per_intent <= 0:
        raise ValueError("examples_per_intent must be a positive integer")

    if isinstance(teacher, (str, Path)):
        teacher = LocalTeacher(teacher)
    if not hasattr(teacher, "generate_examples"):
        raise TypeError("teacher must be a local checkpoint path or provide generate_examples()")
    chosen = set(specialist.intents if intents is None else intents)
    if not chosen or not chosen <= specialist.intents:
        raise ValueError("Choose one or more declared intents for generation")
    excluded = set(excluded_examples or {})
    blocked_hashes = set()
    if specialist._last_trained is not None:
        report = _read_json(specialist._last_trained.output_dir / "training.json")
        blocked_hashes = set(report["split_text_sha256"]["validation"]) | set(
            report["split_text_sha256"]["test"]
        )
        excluded.update(text for text in specialist.examples if _text_hash(text) in blocked_hashes)
    for pointer in _root(specialist).glob("*/latest.json"):
        record = _read_json(pointer)
        report = _read_json(pointer.parent / record["run"] / "training.json")
        blocked_hashes.update(report["split_text_sha256"]["validation"])
        blocked_hashes.update(report["split_text_sha256"]["test"])
    values = teacher.generate_examples(
        intents=sorted(chosen),
        examples=dict(specialist.examples),
        examples_per_intent=examples_per_intent,
        excluded_texts=sorted(excluded),
    )
    proposed = validate_suggestions(values, chosen, specialist.examples, excluded)
    return {
        text: intent for text, intent in proposed.items() if _text_hash(text) not in blocked_hashes
    }
