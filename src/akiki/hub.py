"""Explicit, optional Hugging Face downloads for local MiniLM inference."""

import os
from pathlib import Path

_CHECKPOINT_FILES = ("config.json", "model.safetensors", "tokenizer.json")
_MODEL_FILES = (
    *_CHECKPOINT_FILES,
    *(f"base_model/{name}" for name in _CHECKPOINT_FILES),
    "adapter.safetensors",
    "model_config.json",
    "README.md",
    "LICENSE",
    "LICENSE.*",
    "NOTICE",
    "NOTICE.*",
)


def download_model(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Path:
    """Download or reuse a Hugging Face MiniLM checkpoint and return its local path.

    Requires ``pip install 'akiki[huggingface]'``. Supports plain checkpoints
    and Akiki Ariadne variants with ``base_model/`` and an adapter. Only the
    supported model files, model card, and license notices are selected; other
    weight formats, training data, and repository code are not downloaded.

    ``revision`` selects a branch, tag, or full commit hash (default: main).
    ``cache_dir`` selects a Hub cache; otherwise Hugging Face's configured
    cache is used. ``local_dir`` optionally writes the files to a chosen
    directory instead of returning a snapshot directory in the Hub cache.

    Set ``local_files_only=True`` to require an existing local snapshot.
    ``HF_HUB_OFFLINE`` also forces local-only access and is never overridden.
    A partial snapshot missing required files raises ``FileNotFoundError``.
    With ``local_dir`` in offline mode, its required files must already exist.
    This function does not select a different checkpoint if files are missing.

    ``token`` accepts an access token, ``True`` to require a saved token, or
    ``False`` to disable authentication. The default uses Hugging Face's
    configured authentication, including ``HF_TOKEN`` or a saved login.

    This function may access the network when explicitly called. Ordinary
    ``MiniLM(path)`` construction and inference remain local.

    Examples
    --------
    Use an existing cached checkpoint without network access::

        >>> import os
        >>> os.environ["HF_HUB_OFFLINE"] = "1"
        >>> path = download_model(  # doctest: +SKIP
        ...     "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        ... )
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face support requires the optional dependency: "
            "pip install 'akiki[huggingface]'"
        ) from error

    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=Path(cache_dir).expanduser() if cache_dir is not None else None,
            local_dir=Path(local_dir).expanduser() if local_dir is not None else None,
            local_files_only=local_files_only or offline,
            token=token,
            allow_patterns=list(_MODEL_FILES),
        )
    except Exception as error:
        # Hub HTTP exception types differ across supported dependency versions.
        raise RuntimeError(f"Could not resolve Hugging Face model {repo_id!r}: {error}") from error

    root = Path(snapshot)
    if (root / "base_model").is_dir() and all(
        (root / name).is_file() for name in _CHECKPOINT_FILES
    ):
        raise ValueError(
            "Hugging Face snapshot contains both a root checkpoint and base_model/; "
            "use a clean local_dir to avoid mixing models"
        )
    checkpoint = root / "base_model" if (root / "base_model").is_dir() else root
    required = [checkpoint / name for name in _CHECKPOINT_FILES]
    if checkpoint != root:
        required.append(root / "adapter.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Hugging Face snapshot is missing required MiniLM files: " + ", ".join(missing)
        )
    return root
