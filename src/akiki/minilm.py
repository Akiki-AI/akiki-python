"""Compatibility imports for the text inference API in ``akiki.decision``.

Use ``akiki.decision`` for new code. Both modules expose the same ``MiniLM``,
``Encoding``, ``Classification``, and current text ``ModelPrecision`` objects.
Older precision strings must be replaced with text precision enum members.
"""

from .decision import Classification as Classification
from .decision import Encoding as Encoding
from .decision import MiniLM as MiniLM
from .decision import ModelPrecision as ModelPrecision
from .decision import _load_library as _load_library

__all__ = ["Classification", "Encoding", "MiniLM", "ModelPrecision"]
