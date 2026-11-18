"""Check the grouped public API and its existing import aliases."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import akiki
from akiki import audio, decision
from akiki.audio import vad
from akiki.hub import download_model


@pytest.mark.parametrize(
    "name,expected",
    [
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


@pytest.mark.parametrize("access", ["from akiki import VAD", "VAD = akiki.VAD"])
def test_legacy_top_level_vad_warns_and_keeps_its_identity(access):
    namespace = {"akiki": akiki}
    with pytest.warns(DeprecationWarning, match=r"akiki\.VAD .* removed in version 0\.3\.0"):
        exec(access, namespace)
    assert namespace["VAD"] is vad.VAD


@pytest.mark.parametrize(
    "access",
    [
        "import akiki.vad as legacy",
        "from akiki import vad as legacy",
        "legacy = akiki.vad",
        'legacy = importlib.import_module("akiki.vad")',
    ],
)
def test_legacy_vad_module_warns_and_keeps_its_exports(access):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    import akiki
    from akiki.audio import vad

with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", DeprecationWarning)
    {access}
[warning] = recorded
assert warning.filename == "<string>", warning.filename
assert warning.category is DeprecationWarning
assert "akiki.vad is deprecated and will be removed in version 0.3.0" in str(warning.message)
assert legacy is importlib.import_module("akiki.vad")
for name in vad.__all__:
    assert getattr(legacy, name) is getattr(vad, name)
""",
        ],
        env=os.environ | {"PYTHONPATH": str(Path(akiki.__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", importlib.import_module("akiki.minilm").__all__)
def test_legacy_minilm_imports_keep_their_identity(name):
    legacy = importlib.import_module("akiki.minilm")
    assert getattr(legacy, name) is getattr(decision, name)


def test_wildcard_import_exposes_the_main_interfaces_and_modules():
    namespace = {}
    with pytest.warns(DeprecationWarning, match=r"akiki\.VAD"):
        exec("from akiki import *", namespace)
    assert namespace["VAD"] is vad.VAD
    assert namespace["MiniLM"] is decision.MiniLM
    assert namespace["audio"] is audio
    assert namespace["audio"].vad is vad
    assert "vad" not in namespace
    assert namespace["decision"] is decision
    assert "minilm" not in namespace
    assert "ModelPrecision" not in namespace
    assert "SpeechFrame" not in namespace
    assert "Encoding" not in namespace
    assert "Classification" not in namespace
    assert "training" not in namespace
