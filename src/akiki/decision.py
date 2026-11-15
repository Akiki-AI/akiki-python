"""Local specialist classification and MiniLM text inference.

``MiniLM`` loads a local plain checkpoint or Ariadne variant into the Rust
engine. ``encode`` returns an ``Encoding``; ``classify`` returns a
``Classification`` with label probabilities. Configure numeric formats with
this module's ``ModelPrecision`` enum, which is distinct from VAD's enum.
MiniLM weights are not bundled.

``DecisionModel`` classifies text with saved specialists through the native
MiniLM engine, loading one model/adapter handle per specialist on first use.
Its ``base_model`` property selects the local frozen checkpoint.
``list_base_models()`` lists existing encoder snapshots in local Hugging Face
caches as ``BaseModel`` descriptors, including input embedding table dimensions.
Selection and discovery read metadata without loading model weights.

Examples
--------
>>> from akiki.decision import MiniLM, ModelPrecision
>>> with MiniLM(
...     "path/to/local/MiniLM", model_precision=ModelPrecision.FP32
... ) as model:  # doctest: +SKIP
...     vector = model.embed("A sentence to encode.")
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import threading
import weakref
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from ._affine_cache import AffineLayer, _AffineCache
from ._base_models import BaseModel, _resolve_base_model, list_base_models
from ._engines import library_candidates
from ._specialist_reports import IntentEvaluation, SpecialistEvaluation, TrainingAdvice
from ._specialist_training import (
    TrainedSpecialist,
    _current_artifact,
    evaluate_specialist,
    suggest_examples,
    train_specialist,
)

__all__ = [
    "AffineLayer",
    "BaseModel",
    "Classification",
    "DecisionModel",
    "Encoding",
    "IntentEvaluation",
    "MiniLM",
    "ModelPrecision",
    "Specialist",
    "SpecialistEvaluation",
    "TrainedSpecialist",
    "TrainingAdvice",
    "list_base_models",
]


@dataclass(eq=False)
class Specialist:
    """Intent labels and text-to-intent examples for one specialist.

    Each instance starts with its own empty collections. Specialists compare
    by identity so editing their contents preserves membership in a model's
    ``Specialists`` set.

    ``affine_layers`` is a persistent local mapping keyed by ``BaseModel``.
    Assigning an ``AffineLayer`` saves it atomically; lookups load saved arrays
    without training. Equivalent specialists reuse the cache across sessions.
    Different checkpoint paths, revisions, intents, or examples have separate
    entries. Changing data selects another namespace and preserves old files.

    ``cache_dir`` chooses the local affine cache directory. By default, use
    ``affines/`` under ``AKIKI_CACHE_DIR`` or the platform's Akiki cache.
    Constructing a specialist does not create cache directories or load layers.
    """

    intents: set[str] = field(default_factory=set)
    examples: dict[str, str] = field(default_factory=dict)
    affine_layers: MutableMapping[BaseModel, AffineLayer] = field(default_factory=dict, repr=False)
    cache_dir: str | Path | None = field(default=None, repr=False, kw_only=True)
    _last_trained: TrainedSpecialist | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        initial_layers = self.affine_layers
        self.affine_layers = _AffineCache(self)
        self.affine_layers.update(initial_layers)

    def train(
        self,
        base_model: str | Path | BaseModel,
        *,
        config=None,
        validation_examples: dict[str, str] | None = None,
        test_examples: dict[str, str] | None = None,
        continue_training: bool = False,
        resume_from: TrainedSpecialist | str | Path | None = None,
        on_epoch=None,
    ) -> TrainedSpecialist:
        """Reuse a saved fit or train through the optional ``akiki-training`` backend.

        ``continue_training=True`` warm-starts saved affine/head parameters
        with a fresh optimizer and preserves the original validation split.
        Pass ``resume_from`` for an earlier artifact when changing examples
        across sessions. No model assets are downloaded.
        """
        return train_specialist(
            self,
            base_model,
            config=config,
            validation_examples=validation_examples,
            test_examples=test_examples,
            continue_training=continue_training,
            resume_from=resume_from,
            on_epoch=on_epoch,
        )

    def eval(self, base_model: str | Path | BaseModel | None = None) -> SpecialistEvaluation:
        """Report saved held-out per-intent scores, training gradients, and trends."""
        return evaluate_specialist(self, base_model)

    def suggest_examples(
        self,
        teacher,
        *,
        examples_per_intent: int = 5,
        intents: set[str] | None = None,
        excluded_examples: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Propose local-teacher samples for review without changing examples.

        ``teacher`` is an explicit local checkpoint path, ``LocalTeacher``,
        or an object implementing ``generate_examples``. Generated samples
        exclude known training/holdout duplicates. Review them, then add
        accepted entries to ``examples`` before continuing training.
        """
        return suggest_examples(
            self,
            teacher,
            examples_per_intent=examples_per_intent,
            intents=intents,
            excluded_examples=excluded_examples,
        )


class ModelPrecision(Enum):
    """Native text weight and projection formats, distinct from VAD precision.

    Pass a member to ``DecisionModel`` or ``MiniLM``. ``INT8`` is the default;
    ``INT4_G32`` adds the grouped INT4 format that VAD does not support.
    Attention, normalization, and pooling remain FP32 in every format.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT8_FP32_INPUT = "int8-fp32-input"
    INT4 = "int4"
    INT4_G32 = "int4-g32"

    def __str__(self) -> str:
        return self.value


class DecisionModel:
    """Classify text with local specialists through the Rust MiniLM engine.

    ``base_model`` accepts a local checkpoint path, a ``BaseModel`` returned by
    ``list_base_models()``, or an unambiguous cached repository name. It defaults
    to ``None``. Setting it validates local checkpoint metadata and file presence
    without loading weights or contacting Hugging Face. The selected descriptor
    exposes its path and input embedding table dimensions. Dimensions are read
    from the selected checkpoint at runtime, independently of model families.
    Discovery currently lists encoders; explicit selection also accepts other
    local text model families.

    ``Specialists`` is a mutable set of ``Specialist`` instances. ``classify``
    selects one explicitly, or uses the only attached specialist. ``decide``
    returns a separate classification for every attached specialist.

    Inference requires a complete saved training export for the selected base
    and current specialist data. It never starts training. Native MiniLM/BERT
    specialist exports are supported; other families fail explicitly. No
    training package, PyTorch, Transformers, or network access is required.

    Construction loads no runtime. Each specialist's native handle is loaded
    on first use and reused until its saved export or the base selection changes.
    The Rust engine folds its affine into its own embedding table in memory;
    separate specialists currently use separate copies of the base. Files stay
    unchanged. Calls on one DecisionModel are serialized. Use ``close`` or a
    context manager to release loaded handles.
    """

    def __init__(
        self,
        base_model: str | Path | BaseModel | None = None,
        *,
        threads: int | None = None,
        model_precision: ModelPrecision = ModelPrecision.INT8,
        library_path: str | Path | None = None,
    ):
        """Select local files; native options follow ``MiniLM`` defaults."""
        if not isinstance(model_precision, ModelPrecision):
            raise TypeError("model_precision must be an akiki.decision.ModelPrecision member")
        if threads is not None and (
            isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 256
        ):
            raise ValueError("threads must be an integer from 1 to 256")
        self._lock = threading.RLock()
        self._closed = False
        self._base_model: BaseModel | None = None
        self._specialists: set[Specialist] = set()
        self._runtimes: dict[Specialist, tuple[TrainedSpecialist, MiniLM, dict]] = {}
        self._model_options = {
            "threads": threads,
            "model_precision": model_precision,
            "library_path": library_path,
        }
        self.base_model = base_model

    @property
    def base_model(self) -> BaseModel | None:
        """Selected checkpoint descriptor and embedding dimensions, or ``None``."""
        return self._base_model

    @base_model.setter
    def base_model(self, value: str | Path | BaseModel | None) -> None:
        base = None if value is None else _resolve_base_model(value)
        with self._lock:
            self._check_open()
            if base != self._base_model:
                self._release_runtimes()
                self._base_model = base

    @property
    def Specialists(self) -> set[Specialist]:
        """This model's mutable set of specialists, initially empty."""
        return self._specialists

    @Specialists.setter
    def Specialists(self, value: set[Specialist]) -> None:
        if not isinstance(value, set) or not all(isinstance(item, Specialist) for item in value):
            raise TypeError("Specialists must be a set of Specialist instances")
        with self._lock:
            self._check_open()
            self._specialists = value.copy()
            self._prune_runtimes()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DecisionModel is closed")

    def _attached_specialists(self) -> tuple[Specialist, ...]:
        self._check_open()
        self._prune_runtimes()
        if self.base_model is None:
            raise ValueError("Select DecisionModel.base_model before inference")
        specialists = tuple(self._specialists)
        if not specialists:
            raise ValueError("Add a trained Specialist to DecisionModel.Specialists")
        if not all(isinstance(item, Specialist) for item in specialists):
            raise TypeError("Specialists must contain only Specialist instances")
        return specialists

    def _discard_runtime(self, specialist: Specialist) -> None:
        previous = self._runtimes.pop(specialist, None)
        if previous is not None:
            previous[1].close()

    def _prune_runtimes(self) -> None:
        for specialist in tuple(self._runtimes):
            if specialist not in self._specialists:
                self._discard_runtime(specialist)

    def _release_runtimes(self) -> None:
        for specialist in tuple(self._runtimes):
            self._discard_runtime(specialist)

    def _runtime(self, specialist: Specialist) -> MiniLM:
        try:
            artifact = _current_artifact(specialist, self.base_model)
            config = json.loads((artifact.output_dir / "model_config.json").read_text())
            if not isinstance(config, dict) or config.get("labels") != list(artifact.labels):
                raise ValueError(
                    "Saved specialist label order does not match its classification head"
                )
            family = config.get("model_family", "minilm")
            if family != "minilm":
                raise NotImplementedError(
                    "DecisionModel native specialist inference currently supports MiniLM/BERT "
                    f"exports; {family!r} specialist inference is not exposed by the Python binding"
                )
            if (artifact.output_dir / "base_model").resolve() != self.base_model.path:
                raise ValueError(
                    "Saved specialist base_model does not match DecisionModel.base_model"
                )
        except Exception:
            self._discard_runtime(specialist)
            raise
        previous = self._runtimes.get(specialist)
        if previous is not None:
            if previous[0] == artifact and previous[2] == config:
                return previous[1]
            self._discard_runtime(specialist)
        runtime = MiniLM(artifact.output_dir, **self._model_options)
        if runtime.labels != artifact.labels:
            runtime.close()
            raise ValueError("Saved specialist label order does not match its classification head")
        self._runtimes[specialist] = artifact, runtime, config
        return runtime

    def _selected_runtime(self, specialist: Specialist | None) -> MiniLM:
        specialists = self._attached_specialists()
        if specialist is None:
            if len(specialists) != 1:
                raise ValueError("Several specialists are attached; pass specialist explicitly")
            specialist = specialists[0]
        elif not isinstance(specialist, Specialist):
            raise TypeError("specialist must be a Specialist instance")
        elif specialist not in specialists:
            raise ValueError("Add this specialist to DecisionModel.Specialists first")
        return self._runtime(specialist)

    def tokenize(
        self, text: str, special_tokens: bool = True, *, specialist: Specialist | None = None
    ) -> list[int]:
        """Return the selected specialist's native tokenizer IDs without truncation.

        Selection follows ``classify``: omit ``specialist`` when exactly one is
        attached. Tokenization shares its cached native handle and lock; it never
        creates a second encoder or imports a training runtime. Special tokens
        are included unless ``special_tokens=False``.
        """
        _text(text)
        with self._lock:
            return self._selected_runtime(specialist).tokenize(text, special_tokens=special_tokens)

    def classify(self, text: str, *, specialist: Specialist | None = None) -> Classification:
        """Classify verbatim text with one attached specialist's saved adapter.

        Omit ``specialist`` when exactly one is attached. Results include its
        predicted label, probabilities and logits in the saved label order.
        Missing/stale exports raise an error; inference never trains or truncates
        text. Re-training through ``Specialist.train`` selects the new export
        automatically on the next call.
        """
        _text(text)
        with self._lock:
            return self._selected_runtime(specialist).classify(text)

    def decide(self, text: str) -> dict[Specialist, Classification]:
        """Run each attached specialist, keeping its labels and probabilities separate.

        Specialists execute sequentially through their own cached native handles.
        The result mapping uses the attached Specialist objects as keys.
        """
        _text(text)
        with self._lock:
            specialists = self._attached_specialists()
            return {item: self._runtime(item).classify(text) for item in specialists}

    def close(self) -> None:
        """Release all loaded native handles. Repeated calls have no effect."""
        with self._lock:
            self._release_runtimes()
            self._closed = True

    def __enter__(self) -> DecisionModel:
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_ENGINE_PRECISIONS = {
    ModelPrecision.FP32: "f32",
    ModelPrecision.FP16: "f16",
    ModelPrecision.BF16: "bf16",
    ModelPrecision.INT8: "int8",
    ModelPrecision.INT8_FP32_INPUT: "int8-f32",
    ModelPrecision.INT4: "int4",
    ModelPrecision.INT4_G32: "int4-g32",
}
DEFAULT_THREADS = 4


@dataclass(frozen=True, eq=False)
class Encoding:
    """Engine outputs for one input, returned by ``MiniLM.encode``.

    Attributes
    ----------
    embedding : numpy.ndarray
        L2-normalized float32 vector of length ``MiniLM.embedding_size``.
    logits : numpy.ndarray or None
        Raw float32 head outputs of length ``MiniLM.head_size``, or ``None``
        when the model has no adapter head.
    """

    embedding: np.ndarray
    logits: np.ndarray | None


@dataclass(frozen=True)
class Classification:
    """Most probable head output, returned by ``MiniLM.classify``.

    Attributes
    ----------
    index : int
        Position of the largest logit.
    label : str or None
        Name of that output from the model's labels, or ``None`` without them.
    probabilities : tuple[float, ...]
        Softmax of all logits, in output order. No calibration is applied.
    logits : tuple[float, ...]
        Raw head outputs, in output order.

    Examples
    --------
    >>> result = Classification(1, "question", (0.2, 0.8), (0.0, 1.386))
    >>> result.label, result.probabilities[result.index]
    ('question', 0.8)
    """

    index: int
    label: str | None
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]


def _load_library(library_path=None):
    """Return the first local MiniLM shared library that binds, and its path."""
    failures = []
    for candidate in dict.fromkeys(
        library_candidates(library_path, "akiki_minilm_engine", "AKIKI_MINILM_LIBRARY")
    ):
        if not candidate.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(candidate.resolve()))
            _bind(lib)
        except (OSError, AttributeError) as error:
            failures.append(f"{candidate}: {error}")
            continue
        return lib, candidate.resolve()
    reason = "; ".join(failures) or "no local shared library found"
    if sys.platform == "linux" and platform.machine().lower() in {"aarch64", "arm64"}:
        reason += (
            "; ARM64 MiniLM support is experimental; use a native ARM64 wheel or local library"
        )
    raise RuntimeError(
        f"No compatible MiniLM engine: {reason}. "
        "Use a native akiki wheel or set AKIKI_MINILM_LIBRARY to a local library"
    )


def _bind(lib):
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    bindings = {
        "create": (
            [ctypes.c_char_p, ctypes.c_char_p, size, ctypes.c_char_p, ctypes.c_char_p, size],
            pointer,
        ),
        "free": ([pointer], None),
        "max_tokens": ([pointer], size),
        "embedding_size": ([pointer], size),
        "head_size": ([pointer], size),
        "model_bytes": ([pointer], size),
        "tokenize": (
            [pointer, ctypes.c_char_p, size, ctypes.c_int32, pointer, size],
            ctypes.c_ssize_t,
        ),
        "encode": (
            [pointer, pointer, pointer, size, pointer, pointer, size, pointer, size],
            ctypes.c_int32,
        ),
    }
    for name, (args, result) in bindings.items():
        fn = getattr(lib, "akiki_minilm_" + name)
        fn.argtypes, fn.restype = args, result


def _text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.encode("utf-8")


class MiniLM:
    """Encode text with a local MiniLM model: a plain checkpoint or an Ariadne variant.

    A plain checkpoint, such as ``sentence-transformers/all-MiniLM-L6-v2``,
    produces sentence embeddings. Ariadne variants add an adapter to the frozen
    encoder: an input affine on the word embeddings, a linear head, or both.
    Every model returns an L2-normalized embedding; models with a head also
    return logits, which ``classify`` turns into probabilities. Model weights
    are not bundled with the package. Pass the path to a local copy, or
    explicitly fetch a Hugging Face model with ``MiniLM.from_pretrained``.

    Inference runs on the native Rust engine, which is included in Linux
    x86_64 and ARM64 wheels. ARM64 MiniLM support is experimental. There is no
    ONNX or PyTorch fallback. INT8 projections are the default; see ``MiniLM.__init__`` for all options.

    Use the model as a context manager or call ``close()`` when finished. One
    instance can be shared between threads; its calls run one at a time.

    Attributes
    ----------
    model_path : pathlib.Path
        Checkpoint folder holding the encoder and its tokenizer.
    adapter_path : pathlib.Path or None
        Adapter safetensors file, or ``None`` for a plain checkpoint.
    labels : tuple[str, ...] or None
        Names of the head outputs from ``model_config.json``, if it lists them.
    embedding_size : int
        Length of every embedding.
    head_size : int
        Number of head outputs; zero without an adapter head.
    max_tokens : int
        Longest sequence the encoder accepts, including special tokens.
    chunk_tokens : int or None
        Longest chunk, including special tokens, when long text is split.
        ``None`` means text is encoded as one sequence.
    model_precision : ModelPrecision
        Format of the word embeddings and linear projections.
    threads : int
        CPU worker threads used by this instance.
    library_path : pathlib.Path
        Native engine library that was loaded.
    model_bytes : int
        Resident size of the loaded weights, in bytes.

    Examples
    --------
    Compare sentences with a plain checkpoint::

        >>> from akiki import MiniLM
        >>> with MiniLM("all-MiniLM-L6-v2") as model:  # doctest: +SKIP
        ...     vectors = model.embed(["A man is eating food.", "Someone is eating a meal."])
        >>> round(float(vectors[0] @ vectors[1]), 2)  # doctest: +SKIP
        0.78

    Classify with an Ariadne variant that has a head::

        >>> with MiniLM("Akiki-MiniLM-L6-v2-clover-intent") as model:  # doctest: +SKIP
        ...     model.classify("Current user message: Restart my web server.").label
        'task'
    """

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        **model_options,
    ) -> MiniLM:
        """Fetch a Hugging Face model, then load it into the local Rust engine.

        Requires the optional ``akiki[huggingface]`` extra. ``repo_id`` must
        contain a supported plain MiniLM checkpoint or Akiki Ariadne variant;
        this method does not make arbitrary Hub architectures compatible.

        ``revision``, ``cache_dir``, ``local_dir``, ``local_files_only``, and
        ``token`` follow ``akiki.download_model``. All other options, such as
        ``model_precision``, ``threads``, ``adapter_path``, and ``chunk_tokens``,
        are passed to the local constructor. ``HF_HUB_OFFLINE`` is respected.

        Downloads happen only when this method is explicitly called. To load
        from a local path without Hub access, use ``MiniLM(path)`` instead.

        Examples
        --------
        >>> with MiniLM.from_pretrained(  # doctest: +SKIP
        ...     "sentence-transformers/all-MiniLM-L6-v2"
        ... ) as model:
        ...     vector = model.embed("A man is eating food.")
        """
        from .hub import download_model

        path = download_model(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls(path, **model_options)

    def __init__(
        self,
        model_path: str | Path,
        adapter_path: str | Path | None = None,
        *,
        threads: int | None = None,
        model_precision: ModelPrecision = ModelPrecision.INT8,
        chunk_tokens: int | None = None,
        library_path: str | Path | None = None,
    ):
        """Load a local model into the native engine.

        Parameters
        ----------
        model_path : str or os.PathLike
            A checkpoint folder containing ``config.json``,
            ``model.safetensors``, and ``tokenizer.json``; or a variant folder
            containing such a checkpoint in ``base_model/``. A variant folder's
            ``adapter.safetensors`` is loaded automatically, and its optional
            ``model_config.json`` may name the head outputs (``labels``) and a
            chunk length (``max_length``).
        adapter_path : str or os.PathLike, optional
            Adapter safetensors file to apply, overriding the variant folder's.
            To load a variant's plain encoder, pass its ``base_model`` folder.
        threads : int, optional
            CPU worker threads, from 1 to 256. Defaults to 4, or fewer if
            fewer CPUs are available to this process.
        model_precision : ModelPrecision, default=ModelPrecision.INT8
            Format of the word embeddings and linear projections. Pass a member
            of ``akiki.decision.ModelPrecision``: ``FP32``, ``FP16``, ``BF16``,
            ``INT8``, ``INT8_FP32_INPUT``, ``INT4``, or ``INT4_G32``. Weights are
            converted in memory when loading; the files are unchanged.
            Attention, normalization, and pooling remain FP32. Strings and
            members of ``akiki.vad.ModelPrecision`` are rejected.
        chunk_tokens : int, optional
            Split text into chunks of at most this many tokens, including two
            special tokens, as Ariadne retrieval models do. Defaults to the
            variant's ``max_length``; otherwise text is one sequence.
        library_path : str or os.PathLike, optional
            Path to a specific Rust MiniLM shared library. Overrides
            ``AKIKI_MINILM_LIBRARY`` and automatic discovery.

        Raises
        ------
        TypeError
            ``model_precision`` is not an ``akiki.decision.ModelPrecision`` member.
        FileNotFoundError
            A required model file is missing.
        ValueError
            An option is invalid, or the files are not a compatible MiniLM
            checkpoint and adapter.
        RuntimeError
            No compatible native engine library is available.
        """
        if not isinstance(model_precision, ModelPrecision):
            raise TypeError("model_precision must be an akiki.decision.ModelPrecision member")
        if threads is None:
            available = (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            )
            threads = min(DEFAULT_THREADS, available or 1)
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 256:
            raise ValueError("threads must be an integer from 1 to 256")

        root = Path(model_path).expanduser()
        self.model_path = root / "base_model" if (root / "base_model").is_dir() else root
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (self.model_path / name).is_file():
                raise FileNotFoundError(
                    f"MiniLM checkpoint file is missing: {self.model_path / name}"
                )
        if adapter_path is not None:
            self.adapter_path = Path(adapter_path).expanduser()
            if not self.adapter_path.is_file():
                raise FileNotFoundError(f"MiniLM adapter is missing: {self.adapter_path}")
        else:
            default = root / "adapter.safetensors"
            self.adapter_path = default if default.is_file() else None
        config = {}
        if (root / "model_config.json").is_file():
            config = json.loads((root / "model_config.json").read_text())
        if chunk_tokens is None and isinstance(config.get("max_length"), int):
            chunk_tokens = config["max_length"]

        lib, self.library_path = _load_library(library_path)
        error = ctypes.create_string_buffer(1024)
        adapter = str(self.adapter_path.resolve()).encode("utf-8") if self.adapter_path else None
        handle = lib.akiki_minilm_create(
            str(self.model_path.resolve()).encode("utf-8"),
            adapter,
            threads,
            _ENGINE_PRECISIONS[model_precision].encode("ascii"),
            error,
            len(error),
        )
        if not handle:
            raise ValueError(
                f"MiniLM engine cannot load {root}: {error.value.decode('utf-8', 'replace')}"
            )
        self._lib, self._handle = lib, handle
        self._finalizer = weakref.finalize(self, lib.akiki_minilm_free, handle)
        self._lock = threading.Lock()
        self.model_precision = model_precision
        self.threads = threads
        self.max_tokens = lib.akiki_minilm_max_tokens(handle)
        self.embedding_size = lib.akiki_minilm_embedding_size(handle)
        self.head_size = lib.akiki_minilm_head_size(handle)
        self.model_bytes = lib.akiki_minilm_model_bytes(handle)
        self._ids = np.empty(self.max_tokens + 1, dtype=np.uint32)

        labels = config.get("labels")
        if labels is not None and len(labels) != self.head_size:
            self.close()
            raise ValueError(
                f"model_config.json lists {len(labels)} labels for {self.head_size} head outputs"
            )
        self.labels = tuple(labels) if labels is not None else None
        if chunk_tokens is not None:
            if (
                isinstance(chunk_tokens, bool)
                or not isinstance(chunk_tokens, int)
                or not 3 <= chunk_tokens <= self.max_tokens
            ):
                self.close()
                raise ValueError(f"chunk_tokens must be an integer from 3 to {self.max_tokens}")
            special = self.tokenize("")
            if len(special) != 2:
                self.close()
                raise ValueError("Chunking needs a tokenizer that wraps text in two special tokens")
            self._cls, self._sep = special
        self.chunk_tokens = chunk_tokens

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("MiniLM model is closed")

    def _tokenize(self, data, special_tokens):
        ids = self._ids
        count = self._lib.akiki_minilm_tokenize(
            self._handle, data, len(data), int(special_tokens), ids.ctypes.data, len(ids)
        )
        if count > len(ids):
            ids = np.empty(count, dtype=np.uint32)
            count = self._lib.akiki_minilm_tokenize(
                self._handle, data, len(data), int(special_tokens), ids.ctypes.data, len(ids)
            )
        if count < 0:
            raise RuntimeError("MiniLM tokenizer failed")
        return ids[:count].copy()

    def _encode(self, chunks, weights):
        ids = np.ascontiguousarray(np.concatenate(chunks), dtype=np.uint32)
        lengths = np.array([len(chunk) for chunk in chunks], dtype=np.uintp)
        weights = None if weights is None else np.ascontiguousarray(weights, dtype=np.float32)
        embedding = np.empty(self.embedding_size, dtype=np.float32)
        logits = np.empty(self.head_size, dtype=np.float32) if self.head_size else None
        status = self._lib.akiki_minilm_encode(
            self._handle,
            ids.ctypes.data,
            lengths.ctypes.data,
            len(chunks),
            None if weights is None else weights.ctypes.data,
            embedding.ctypes.data,
            len(embedding),
            None if logits is None else logits.ctypes.data,
            0 if logits is None else len(logits),
        )
        if status != 0:
            raise ValueError(
                f"Each chunk needs 1 to {self.max_tokens} valid token IDs, including special "
                "tokens, and a positive finite weight"
            )
        return Encoding(embedding, logits)

    def tokenize(self, text: str, special_tokens: bool = True) -> list[int]:
        """Return the checkpoint tokenizer's IDs for ``text``, never truncated or padded.

        Parameters
        ----------
        text : str
            Text to tokenize.
        special_tokens : bool, default True
            Whether to add the tokenizer's special tokens, such as ``[CLS]``
            and ``[SEP]``.
        """
        data = _text(text)
        with self._lock:
            self._check_open()
            return self._tokenize(data, special_tokens).tolist()

    def encode(self, text: str) -> Encoding:
        """Return the embedding and any head logits for ``text``.

        Without ``chunk_tokens``, the text is one sequence of at most
        ``max_tokens`` tokens. With it, the text's tokens are split into
        chunks, each wrapped in the special tokens. The embedding is the
        normalized mean of the chunks' normalized embeddings, weighted by their
        text tokens, and the head is applied to it.

        Raises
        ------
        TypeError
            ``text`` is not a string.
        ValueError
            Without ``chunk_tokens``, the text exceeds ``max_tokens`` tokens.
            Inputs are never truncated.
        RuntimeError
            The model has been closed, or the engine failed.
        """
        data = _text(text)
        with self._lock:
            self._check_open()
            if self.chunk_tokens is None:
                ids = self._tokenize(data, True)
                if len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Input has {len(ids)} tokens, over the {self.max_tokens}-token limit "
                        "including special tokens; it is not truncated. Set chunk_tokens to "
                        "encode long text in chunks"
                    )
                return self._encode([ids], None)
            tokens = self._tokenize(data, False)
            room = self.chunk_tokens - 2
            pieces = [tokens[i : i + room] for i in range(0, len(tokens), room)] or [tokens]
            chunks = [np.concatenate(([self._cls], piece, [self._sep])) for piece in pieces]
            return self._encode(chunks, [max(1, len(piece)) for piece in pieces])

    def encode_chunks(self, chunks, weights) -> Encoding:
        """Combine prepared token sequences into one embedding and any logits.

        Use this for variants whose inputs need task-specific chunks, such as
        typed-decision questions repeated in every chunk of a long state.

        Parameters
        ----------
        chunks : sequence of sequence of int
            Complete token sequences, each including its special tokens and
            at most ``max_tokens`` long.
        weights : sequence of float
            One positive weight per chunk, typically its count of text tokens.

        Returns
        -------
        Encoding
            The normalized weighted mean of the chunks' normalized embeddings,
            and the head applied to it.
        """
        arrays = [np.asarray(chunk) for chunk in chunks]
        weights = [float(w) for w in weights]
        if not arrays or len(arrays) != len(weights):
            raise ValueError("Supply one weight for each of at least one chunk")
        for array in arrays:
            if (
                array.ndim != 1
                or not array.size
                or not np.issubdtype(array.dtype, np.integer)
                or array.min() < 0
                or array.max() > np.iinfo(np.uint32).max
            ):
                raise ValueError("Each chunk must be a nonempty sequence of token IDs")
        with self._lock:
            self._check_open()
            return self._encode(arrays, weights)

    def embed(self, texts):
        """Return L2-normalized embeddings; their dot products are cosine similarities.

        Parameters
        ----------
        texts : str or sequence of str
            One text, or several.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(embedding_size,)`` for one text, or
            ``(len(texts), embedding_size)`` for several.
        """
        if isinstance(texts, str):
            return self.encode(texts).embedding
        vectors = [self.encode(text).embedding for text in texts]
        return (
            np.stack(vectors) if vectors else np.empty((0, self.embedding_size), dtype=np.float32)
        )

    def classify(self, text: str) -> Classification:
        """Return the most probable head output for ``text``, with probabilities.

        The text must already be in the variant's input format. Probabilities
        are a softmax over all head outputs; variants that use only some
        outputs per input should mask ``encode(text).logits`` instead.

        Raises
        ------
        ValueError
            The model has no adapter head, or ``encode`` rejects the text.
        """
        if not self.head_size:
            raise ValueError("This model has no classification head; use embed()")
        logits = self.encode(text).logits
        if not np.isfinite(logits).all():
            raise RuntimeError("MiniLM engine returned non-finite logits")
        weights = np.exp(logits.astype(np.float64) - logits.max())
        index = int(np.argmax(logits))
        return Classification(
            index,
            self.labels[index] if self.labels else None,
            tuple((weights / weights.sum()).tolist()),
            tuple(logits.tolist()),
        )

    def close(self) -> None:
        """Release the model and its worker threads; safe to call repeatedly."""
        with self._lock:
            self._finalizer()

    def __enter__(self):
        """Return this open model for use in a ``with`` block."""
        self._check_open()
        return self

    def __exit__(self, *_):
        """Close the model on leaving a ``with`` block; propagate exceptions."""
        self.close()
