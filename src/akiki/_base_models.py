"""Discover local encoders and inspect text checkpoint metadata."""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_LOCAL_HUB_CACHE = Path("/mnt/models/hub")


@dataclass(frozen=True)
class BaseModel:
    """A local text checkpoint and its input token embedding table dimensions.

    ``path`` is the absolute checkpoint directory. ``vocab_size`` is the
    number of rows in its input embedding table and ``embedding_dim`` is the
    width of each token vector. A width-preserving Ariadne input affine uses
    an ``(embedding_dim, embedding_dim)`` weight matrix and an
    ``(embedding_dim,)`` bias, separate from any classification head.
    """

    path: Path
    vocab_size: int
    embedding_dim: int

    @property
    def name(self) -> str:
        """Cached repository name, or the directory name for a local checkpoint."""
        repository = self.path.parent.parent.name
        if self.path.parent.name == "snapshots" and repository.startswith("models--"):
            return repository.removeprefix("models--").replace("--", "/")
        return self.path.name

    @property
    def embedding_shape(self) -> tuple[int, int]:
        """Input embedding table shape as ``(vocab_size, embedding_dim)``."""
        return self.vocab_size, self.embedding_dim

    def __fspath__(self) -> str:
        """Allow the descriptor to be passed to filesystem APIs as its path."""
        return str(self.path)


class _EncoderType(Enum):
    """Recognized encoder families in serialized Hugging Face configurations."""

    ALBERT = "albert"
    BERT = "bert"
    CAMEMBERT = "camembert"
    DEBERTA = "deberta"
    DEBERTA_V2 = "deberta-v2"
    DISTILBERT = "distilbert"
    ELECTRA = "electra"
    EMBEDDING_GEMMA = "embedding_gemma"
    EMBEDDING_GEMMA2 = "embedding_gemma2"
    ERNIE = "ernie"
    LONGFORMER = "longformer"
    MOBILEBERT = "mobilebert"
    MODERNBERT = "modernbert"
    MPNET = "mpnet"
    ROBERTA = "roberta"
    XLM_ROBERTA = "xlm-roberta"


def _cache_roots(cache_dir: str | Path | None) -> list[Path]:
    if cache_dir is not None:
        return [Path(cache_dir).expanduser().resolve()]
    roots = [_LOCAL_HUB_CACHE]
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if value := os.environ.get(variable):
            roots.append(Path(value))
    default_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"
    roots.append(Path(os.environ.get("HF_HOME", str(default_home))) / "hub")
    return list(dict.fromkeys(root.expanduser().resolve() for root in roots))


def _read_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return metadata


def _has_weights(path: Path) -> bool:
    if any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if not (path / name).is_file():
            continue
        weights = _read_metadata(path / name).get("weight_map")
        if isinstance(weights, dict) and weights:
            if all(
                isinstance(shard, str) and (path / shard).is_file() for shard in weights.values()
            ):
                return True
    return False


def _is_input_embedding(name: str) -> bool:
    return any(
        name == suffix or name.endswith("." + suffix)
        for suffix in ("word_embeddings.weight", "tok_embeddings.weight", "embed_tokens.weight")
    )


def _checked_embedding_shape(shape) -> tuple[int, int]:
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError(f"Invalid input embedding table shape: {shape!r}")
    return tuple(shape)


def _safetensors_embedding_shapes(path: Path, config: dict) -> set[tuple[int, int]]:
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        size = int.from_bytes(size_bytes, "little")
        # Bound metadata reads and never read tensor data.
        if len(size_bytes) != 8 or not 0 < size <= min(
            100_000_000, os.fstat(stream.fileno()).st_size - 8
        ):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    shapes = set()
    for name, tensor in header.items():
        if _is_input_embedding(name):
            if not isinstance(tensor, dict):
                raise ValueError(f"Invalid input embedding tensor metadata: {path}")
            shapes.add(_checked_embedding_shape(tensor.get("shape")))
    if not shapes:
        # Architectures can name their parameters freely. A table indexed by
        # the configured vocabulary identifies candidate dimensions even when
        # its parameter name is unfamiliar. Conflicting shapes stay ambiguous.
        text = config.get("text_config", config)
        vocab_size = text.get("vocab_size") if isinstance(text, dict) else None
        for tensor in header.values():
            shape = tensor.get("shape") if isinstance(tensor, dict) else None
            if isinstance(shape, list) and len(shape) == 2 and shape[0] == vocab_size:
                shapes.add(_checked_embedding_shape(shape))
    return shapes


def _embedding_shape(path: Path, config: dict) -> tuple[int, int]:
    shapes = set()
    if (path / "model.safetensors").is_file():
        shapes.update(_safetensors_embedding_shapes(path / "model.safetensors", config))
    elif (path / "model.safetensors.index.json").is_file():
        weights = _read_metadata(path / "model.safetensors.index.json").get("weight_map")
        if (
            not isinstance(weights, dict)
            or not weights
            or not all(isinstance(shard, str) for shard in weights.values())
        ):
            raise ValueError(f"Invalid safetensors weight index: {path}")
        shards = {shard for name, shard in weights.items() if _is_input_embedding(name)}
        if not shards:
            shards = set(weights.values())
        for shard in sorted(shards):
            shapes.update(_safetensors_embedding_shapes(path / shard, config))
    if len(shapes) > 1:
        raise ValueError(f"Multiple input embedding table dimensions in {path}: {sorted(shapes)}")
    if shapes:
        return shapes.pop()
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError(f"Invalid text configuration in {path}")
    width = next(
        (
            text[key]
            for key in ("embedding_size", "hidden_size", "dim", "d_model")
            if text.get(key) is not None
        ),
        None,
    )
    return _checked_embedding_shape((text.get("vocab_size"), width))


def _is_encoder(config: dict) -> bool:
    """Keep the initial discovery list limited to recognized encoder configs."""
    try:
        _EncoderType(config.get("model_type"))
    except ValueError:
        return False
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list) or not all(isinstance(a, str) for a in architectures):
        return False
    return not (
        config.get("is_decoder")
        or config.get("is_encoder_decoder")
        or any(
            marker in architecture
            for architecture in architectures
            for marker in ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")
        )
    )


def _inspect_base_model(path: Path, config: dict | None = None) -> BaseModel:
    """Read dimensions from a checkpoint without restricting its model family."""
    path = path.expanduser().resolve()
    if config is None:
        config = _read_metadata(path / "config.json")
    tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "vocab.txt", "spiece.model", "sentencepiece.bpe.model")
    ) or all((path / name).is_file() for name in ("vocab.json", "merges.txt"))
    if not tokenizer:
        raise FileNotFoundError(f"Tokenizer files are missing: {path}")
    if not _has_weights(path):
        raise FileNotFoundError(f"Model weights are missing or incomplete: {path}")
    vocab_size, embedding_dim = _embedding_shape(path, config)
    return BaseModel(path, vocab_size, embedding_dim)


def list_base_models(cache_dir: str | Path | None = None) -> list[BaseModel]:
    """Return local encoder descriptors that can be assigned to ``base_model``.

    Check ``/mnt/models/hub``, ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE``, and
    ``HF_HOME/hub`` (default: ``$XDG_CACHE_HOME/huggingface/hub`` or
    ``~/.cache/huggingface/hub``). An explicit ``cache_dir`` searches only that
    cache. Missing caches return an empty list.

    Recognize encoder families from local ``config.json`` metadata, including
    BERT/MiniLM, ModernBERT, RoBERTa, and embedding encoders. Require tokenizer
    files and safetensors or PyTorch weights, including every indexed shard.
    Skip malformed or incomplete snapshots, decoders, and encoder-decoder
    models. Each cached revision has its own path; none is selected for you.

    Each ``BaseModel`` exposes ``path``, ``name``, ``vocab_size``,
    ``embedding_dim``, and ``embedding_shape``. Input table dimensions come
    from safetensors headers when available, otherwise local configuration.
    Snapshots with missing or invalid dimensions are skipped. Input embedding
    width can differ from the model's hidden or sentence embedding width.

    This inspects files only. It does not load weights, import optional model
    runtimes or the Hub SDK, download assets, or guarantee inference support.
    """
    models = {}
    for cache in _cache_roots(cache_dir):
        for snapshot in cache.glob("models--*/snapshots/*"):
            try:
                config = _read_metadata(snapshot / "config.json")
                if not _is_encoder(config):
                    continue
                model = _inspect_base_model(snapshot, config)
                models.setdefault(model.path, model)
            except (OSError, ValueError):
                continue
    return sorted(models.values(), key=lambda model: (model.name, str(model.path)))


def _resolve_base_model(model: str | Path | BaseModel) -> BaseModel:
    if isinstance(model, BaseModel):
        return _inspect_base_model(model.path)
    if not isinstance(model, (str, Path)):
        raise TypeError(
            "base_model must be a BaseModel, local path, cached repository name, or None"
        )
    path = Path(model).expanduser()
    if isinstance(model, Path) or path.is_absolute() or path.exists():
        return _inspect_base_model(path)
    matches = {}
    repository = "models--" + model.replace("/", "--")
    for cache in _cache_roots(None):
        for snapshot in (cache / repository / "snapshots").glob("*"):
            try:
                entry = _inspect_base_model(snapshot)
                matches.setdefault(entry.path, entry)
            except (OSError, ValueError):
                continue
    matches = sorted(matches.values(), key=lambda entry: str(entry.path))
    if len(matches) == 1:
        return matches[0]
    if matches:
        paths = ", ".join(str(entry.path) for entry in matches)
        raise ValueError(f"Multiple local revisions of {model!r}: {paths}. Pass the desired path.")
    raise FileNotFoundError(
        f"No local checkpoint for {model!r}. Pass its local path or select from "
        "list_base_models(). Models are never downloaded."
    )
