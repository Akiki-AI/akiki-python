"""Compatibility imports for the MiniLM/BERT runtime in ``akiki.decision``.

Use ``akiki.decision.DecisionModel`` for specialist decisions.
Use ``akiki.decision.MiniLM`` for direct access to the supported family.
Both modules expose the same ``MiniLM``,
``Encoding``, ``Classification``, and current text ``ModelPrecision`` objects.
Older precision strings must be replaced with text precision enum members.
"""

from .decision import Classification as Classification
from .decision import Encoding as Encoding
from .decision import MiniLM as MiniLM
from .decision import ModelPrecision as ModelPrecision
from .decision import _load_library as _load_library

__all__ = ["Classification", "Encoding", "MiniLM", "ModelPrecision"]
