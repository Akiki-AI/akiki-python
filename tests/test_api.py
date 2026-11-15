"""Check the grouped public API and its existing import aliases."""

import importlib

import pytest

from akiki import decision, vad
from akiki.hub import download_model


@pytest.mark.parametrize(
    "name,expected",
    [
        ("VAD", vad.VAD),
        ("MiniLM", decision.MiniLM),
        ("SpeechFrame", vad.SpeechFrame),
        ("Engine", vad.Engine),
        ("ModelPrecision", vad.ModelPrecision),
        ("FFTPrecision", vad.FFTPrecision),
        ("Encoding", decision.Encoding),
        ("Classification", decision.Classification),
        ("download_model", download_model),
    ],
)
def test_existing_top_level_imports_keep_their_identity(name, expected):
    namespace = {}
    exec(f"from akiki import {name}", namespace)
    assert namespace[name] is expected


@pytest.mark.parametrize("name", importlib.import_module("akiki.minilm").__all__)
def test_legacy_minilm_imports_keep_their_identity(name):
    legacy = importlib.import_module("akiki.minilm")
    assert getattr(legacy, name) is getattr(decision, name)


def test_wildcard_import_exposes_the_main_interfaces_and_modules():
    namespace = {}
    exec("from akiki import *", namespace)
    assert namespace["VAD"] is vad.VAD
    assert namespace["MiniLM"] is decision.MiniLM
    assert namespace["vad"] is vad
    assert namespace["decision"] is decision
    assert "minilm" not in namespace
    assert "ModelPrecision" not in namespace
    assert "SpeechFrame" not in namespace
    assert "Encoding" not in namespace
    assert "Classification" not in namespace
    assert "training" not in namespace
