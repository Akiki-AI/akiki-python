"""Forward the public API of the optional ``akiki-training`` package.

Use ``train_intents`` for intent classifiers, ``train`` with a configured
trainer, and ``list_local_models`` to inspect existing local checkpoints.
The companion package controls the available training implementations and
supported checkpoint families.

Public attributes, wildcard imports, and ``dir(training)`` follow the current
``akiki_training.__all__`` without maintaining a separate export list here.
The original ``akiki_training.training`` names, such as ``train_classifier``
and ``TrainingConfig``, remain available through named imports and attributes.
They are omitted from the primary browsing view unless the companion's public
API also exports them.

Python caches imported modules. After editing the companion package, restart
the notebook kernel or reload the changed companion modules to see new code.

Install ``akiki[training]``, then use ``from akiki import training``. The trainer
is maintained in the separate akiki-training project. The extra selects its
``ariadne`` runtime dependencies. Importing the wrapper does not load PyTorch
or Transformers. Training and local teacher generation load optional runtimes
when called. Dataset tools and Lab require separate companion packages.

MiniLM/BERT exports in the supported native file layout can run through Akiki's text API.
Other training families supported by the companion do not automatically gain
native inference support in this package.
"""

try:
    import akiki_training as _backend
    import akiki_training.training as _legacy
except ModuleNotFoundError as error:
    if error.name not in {"akiki_training", "akiki_training.training"}:
        raise
    raise ImportError(
        "Akiki training requires the optional akiki-training package. "
        "Install it with: pip install 'akiki[training]'"
    ) from error


def __getattr__(name: str):
    """Resolve current public exports, then original compatibility names."""
    if name == "__all__":
        return _backend.__all__
    if name in _backend.__all__:
        return getattr(_backend, name)
    if name in _legacy.__all__:
        return getattr(_legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the companion package's current public API."""
    return sorted(_backend.__all__)
