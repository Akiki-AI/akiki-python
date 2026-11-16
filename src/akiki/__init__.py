"""Local model inference and training.

The public API is organized into four areas:

``akiki.vad``
    Streaming voice activity detection with ``VAD``. Results are
    ``SpeechFrame`` objects. Configure inference with this module's ``Engine``,
    ``ModelPrecision``, and ``FFTPrecision`` enums. The VAD model is bundled.
``akiki.decision``
    Select a local encoder with ``DecisionModel`` and ``list_base_models``.
    Classify saved specialists with ``DecisionModel`` through the Akiki engine.
    Run text embeddings and classification directly with ``MiniLM``. Results are
    ``Encoding`` and ``Classification`` objects. This module has its own
    ``ModelPrecision`` enum. Pass a local MiniLM checkpoint or Ariadne variant;
    MiniLM weights are not bundled.
``akiki.training``
    Optional classifier training backed by the separate ``akiki-training``
    package. Install ``akiki[training]``, then use ``from akiki import training``.
    Importing Akiki alone does not load the training package or its runtimes.
``akiki.teacher``
    Experimental review of specialist decisions using an OpenAI-compatible
    server. Propose supplemental examples for retraining with ``Teacher``.
    This module is loaded only when explicitly imported.

``VAD`` and ``MiniLM`` are also available directly from ``akiki`` as shortcuts.
Import supporting result types and configuration enums from their respective
modules. VAD and MiniLM precision enums have different supported formats and
are not interchangeable.

``dir(akiki)`` lists the primary classes, modules, and version. ``training`` and
``teacher`` appear there after they are explicitly imported. Compatibility
aliases and internal modules are omitted from this browsing view.

Examples
--------
Process one frame of silence using the top-level shortcut::

    >>> import akiki
    >>> with akiki.VAD() as detector:
    ...     frames = detector.push([0.0] * 256)
    >>> len(frames)
    1
    >>> frames[0].start_seconds, frames[0].end_seconds
    (0.0, 0.016)

Find domain-specific configuration and help through the modules::

    >>> from akiki import decision, vad
    >>> vad.ModelPrecision.INT8 is decision.ModelPrecision.INT8
    False

Use ``help(akiki.vad)`` and ``help(akiki.decision)`` to explore each API, or
``help(akiki.VAD)`` and ``help(akiki.MiniLM)`` for the main classes. Explicit
model retrieval utilities live in ``akiki.hub``.

Compatibility
-------------
``akiki.minilm`` remains a compatibility module. It re-exports the decision
API's class, result types, and current precision enum.
Existing named top-level imports of ``SpeechFrame``, ``Engine``, ``ModelPrecision``,
``FFTPrecision``, ``Encoding``, ``Classification``, and ``download_model`` remain
available. The top-level precision enums refer to VAD.
Wildcard imports expose only the primary names in ``__all__``.
MiniLM requires the separate ``akiki.decision.ModelPrecision`` enum.

Attributes
----------
__version__ : str
    Installed Akiki package version.
"""

from . import decision, vad
from . import minilm as minilm
from .decision import Classification as Classification
from .decision import Encoding as Encoding
from .decision import MiniLM
from .hub import download_model as download_model
from .vad import VAD
from .vad import Engine as Engine
from .vad import FFTPrecision as FFTPrecision
from .vad import ModelPrecision as ModelPrecision
from .vad import SpeechFrame as SpeechFrame

__version__ = "0.2.0"
__all__ = [
    "MiniLM",
    "VAD",
    "__version__",
    "decision",
    "vad",
]


def __dir__() -> list[str]:
    """List the primary API and explicitly imported training/teacher modules."""
    names = __all__.copy()
    for name in ("training", "teacher"):
        if name in globals():
            names.append(name)
    return sorted(names)
