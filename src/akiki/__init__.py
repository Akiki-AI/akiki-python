"""Local voice activity detection, object detection, and decision models.

The primary interfaces are ``akiki.audio.vad.VAD``,
``akiki.vision.object_detection.ObjectDetector``, and ``akiki.decision.DecisionModel``.
Training and teacher review support the decision-model workflow.
The public API has five areas.

``akiki.audio.vad``
    Streaming voice activity detection with ``VAD``. Results are
    ``SpeechFrame`` objects. Select the model with ``Model``. Configure inference
    with ``Engine``, ``ModelPrecision``, and ``FFTPrecision``.
    The default Akiki-VAD model is bundled.
``akiki.vision.object_detection``
    Use ``ObjectDetector`` to find objects in RGB images.
    Each ``Detection`` gives a family, score, and box.
    Load Akiki Cam Seen from its local Hugging Face snapshot.
``akiki.decision``
    Select a local base checkpoint with ``DecisionModel`` and ``list_base_models``.
    Classify saved specialists with ``DecisionModel`` through the Akiki engine.
    Current native inference supports MiniLM/BERT exports. The lower-level
    ``MiniLM`` class gives direct access to embeddings and classification for
    that family. Results are ``Encoding`` and ``Classification`` objects.
    This module has its own ``ModelPrecision`` enum. Base weights are not bundled.
``akiki.training``
    Optional classifier training backed by the private ``akiki-training``
    companion. Install it separately from an authorized local checkout or wheel.
    ``akiki[training]`` installs runtime dependencies only.
    Importing Akiki alone does not load the training package or its runtimes.
``akiki.teacher``
    Experimental review of specialist decisions using an OpenAI-compatible
    server. Propose supplemental examples for retraining with ``Teacher``.
    This module is loaded only when explicitly imported.

``MiniLM`` is also available directly from ``akiki`` as a shortcut.
Import supporting result types and configuration enums from their respective
modules. VAD and decision precision enums have different supported formats and
are not interchangeable.

``dir(akiki)`` lists the primary classes, modules, and version. ``training`` and
``teacher`` appear there after they are explicitly imported. Compatibility
aliases and internal modules are omitted from this browsing view.

Examples
--------
Process one frame of silence::

    >>> from akiki.audio import vad
    >>> with vad.VAD() as detector:
    ...     frames = detector.push([0.0] * 256)
    >>> len(frames)
    1
    >>> frames[0].start_seconds, frames[0].end_seconds
    (0.0, 0.016)

Find domain-specific configuration and help through the modules::

    >>> from akiki import decision
    >>> from akiki.audio import vad
    >>> vad.ModelPrecision.INT8 is decision.ModelPrecision.INT8
    False

Use ``help(akiki.audio.vad)`` and ``help(akiki.decision)`` to explore each API, or
``help(akiki.audio.vad.VAD)`` and ``help(akiki.decision.DecisionModel)`` for the
primary classes. Use ``help(akiki.decision.MiniLM)`` for direct access to the
currently supported decision-model family.
Explicit model retrieval utilities live in ``akiki.hub``.

Compatibility
-------------
``akiki.vad`` and ``akiki.VAD`` will be removed in version 0.3.0.
Use ``from akiki.audio import vad`` and call ``vad.VAD()``.
``akiki.minilm`` remains a compatibility module. It re-exports the decision
API's MiniLM class, result types, and current precision enum.
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

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from . import audio, decision, vision
from . import minilm as minilm
from .audio.vad import Engine as Engine
from .audio.vad import FFTPrecision as FFTPrecision
from .audio.vad import ModelPrecision as ModelPrecision
from .audio.vad import SpeechFrame as SpeechFrame
from .decision import Classification as Classification
from .decision import Encoding as Encoding
from .decision import MiniLM
from .hub import download_model as download_model

if TYPE_CHECKING:
    from .audio.vad import VAD as VAD

__version__ = "0.2.2"
__all__ = [
    "MiniLM",
    "VAD",
    "__version__",
    "audio",
    "decision",
    "vision",
]


def __getattr__(name: str):
    if name == "VAD":
        warnings.warn(
            "akiki.VAD is deprecated and will be removed in version 0.3.0. "
            "Use akiki.audio.vad.VAD instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return audio.vad.VAD
    if name == "vad":
        return import_module(".vad", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the primary API and explicitly imported training/teacher modules."""
    names = __all__.copy()
    for name in ("training", "teacher"):
        if name in globals():
            names.append(name)
    return sorted(names)
