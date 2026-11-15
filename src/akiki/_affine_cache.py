"""Persist trained input affines locally without importing inference runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ._base_models import BaseModel
from ._quantization import default_cache_dir

_CACHE_VERSION = 1


@dataclass(frozen=True, eq=False)
class AffineLayer:
    """Trained input affine parameters, separate from a classification head.

    For an input embedding width of ``d``, ``weight`` has shape ``(d, d)``
    and ``bias`` has shape ``(d,)``. These arrays represent the transform
    ``weight @ token_embedding + bias``.
    """

    weight: np.ndarray
    bias: np.ndarray


class _SpecialistData(Protocol):
    intents: set[str]
    examples: dict[str, str]
    cache_dir: str | Path | None


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _base_identity(base: BaseModel) -> dict:
    if not isinstance(base, BaseModel):
        raise TypeError("affine_layers keys must be BaseModel descriptors")
    if any(type(size) is not int or size <= 0 for size in base.embedding_shape):
        raise ValueError("BaseModel embedding dimensions must be positive integers")
    return {
        "path": str(Path(base.path).expanduser().resolve()),
        "vocab_size": base.vocab_size,
        "embedding_dim": base.embedding_dim,
    }


def _validate_layer(base: BaseModel, layer: AffineLayer) -> None:
    if not isinstance(layer, AffineLayer):
        raise TypeError("affine_layers values must be AffineLayer instances")
    for name, shape in (
        ("weight", (base.embedding_dim, base.embedding_dim)),
        ("bias", (base.embedding_dim,)),
    ):
        array = getattr(layer, name)
        if not isinstance(array, np.ndarray):
            raise TypeError(f"AffineLayer.{name} must be a NumPy array")
        if array.shape != shape:
            raise ValueError(f"AffineLayer.{name} must have shape {shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"AffineLayer.{name} must contain finite floating-point values")


class _AffineCache(MutableMapping[BaseModel, AffineLayer]):
    """A disk-backed mapping scoped to a specialist's current training data."""

    def __init__(self, specialist: _SpecialistData):
        self._specialist = specialist

    def _namespace(self) -> tuple[Path, str]:
        specialist = self._specialist
        if not isinstance(specialist.intents, set) or not all(
            isinstance(intent, str) for intent in specialist.intents
        ):
            raise TypeError("Specialist.intents must be a set of strings")
        if not isinstance(specialist.examples, dict) or not all(
            isinstance(text, str) and isinstance(intent, str)
            for text, intent in specialist.examples.items()
        ):
            raise TypeError("Specialist.examples must map text strings to intent strings")
        fingerprint = _digest(
            {
                "version": _CACHE_VERSION,
                "intents": sorted(specialist.intents),
                "examples": sorted(specialist.examples.items()),
            }
        )
        root = (
            default_cache_dir() / "affines"
            if specialist.cache_dir is None
            else Path(specialist.cache_dir).expanduser()
        )
        return root.resolve() / fingerprint, fingerprint

    def _entry(self, base: BaseModel) -> tuple[Path, dict]:
        identity = _base_identity(base)
        directory, fingerprint = self._namespace()
        return directory / (_digest(identity) + ".npz"), {
            "version": _CACHE_VERSION,
            "specialist": fingerprint,
            "base_model": identity,
        }

    def __getitem__(self, base: BaseModel) -> AffineLayer:
        path, expected = self._entry(base)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "weight", "bias"}:
                    raise ValueError("Missing or unexpected affine cache arrays")
                if json.loads(archive["metadata"].item()) != expected:
                    raise ValueError("Affine cache metadata does not match the requested entry")
                layer = AffineLayer(archive["weight"], archive["bias"])
            _validate_layer(base, layer)
            return layer
        except FileNotFoundError:
            raise KeyError(base) from None
        except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"Cannot read trained affine cache entry {path}: {error}") from error

    def __setitem__(self, base: BaseModel, layer: AffineLayer) -> None:
        path, metadata = self._entry(base)
        _validate_layer(base, layer)
        layer = AffineLayer(layer.weight.copy(), layer.bias.copy())
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".tmp-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez(
                    stream, metadata=np.array(_json(metadata)), weight=layer.weight, bias=layer.bias
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError as error:
            raise RuntimeError(
                f"Cannot save trained affine cache entry {path}. "
                "Set Specialist(cache_dir=...) or AKIKI_CACHE_DIR to a writable directory."
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def __delitem__(self, base: BaseModel) -> None:
        path, _ = self._entry(base)
        try:
            path.unlink()
        except FileNotFoundError:
            raise KeyError(base) from None

    def __iter__(self) -> Iterator[BaseModel]:
        directory, fingerprint = self._namespace()
        for path in sorted(directory.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(archive["metadata"].item())
                identity = metadata["base_model"]
                base = BaseModel(
                    Path(identity["path"]), identity["vocab_size"], identity["embedding_dim"]
                )
                expected_path, expected = self._entry(base)
                if (
                    path != expected_path
                    or metadata != expected
                    or metadata["specialist"] != fingerprint
                ):
                    raise ValueError(
                        "Affine cache metadata does not match its filename or namespace"
                    )
            except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
                raise ValueError(
                    f"Cannot read trained affine cache entry {path}: {error}"
                ) from error
            yield base

    def __len__(self) -> int:
        return sum(1 for _ in self)
