"""Exercise the optional Hub boundary with local synthetic repositories only."""

import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import akiki
from akiki import MiniLM, download_model
from akiki.__main__ import main
from akiki.decision import ModelPrecision


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Stand in for snapshot_download; never import the SDK or contact the Hub."""
    hub = SimpleNamespace(source=None, destination=tmp_path / "snapshot", calls=[], error=None)

    def snapshot_download(**options):
        hub.calls.append(options)
        if hub.error is not None:
            raise hub.error
        if hub.source is not None:
            destination = options["local_dir"] or hub.destination
            destination.mkdir(parents=True, exist_ok=True)
            for source in hub.source.rglob("*"):
                relative = source.relative_to(hub.source)
                if source.is_file() and any(
                    fnmatch.fnmatch(relative.as_posix(), pattern)
                    for pattern in options["allow_patterns"]
                ):
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            return str(destination)
        return str(hub.destination)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    # Exercise explicit network opt-in against the fake boundary only.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return hub


def test_huggingface_dependency_is_optional_and_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match=r"pip install 'akiki\[huggingface\]'"):
        download_model("org/model")


@pytest.mark.parametrize("layout", ["plain", "variant"])
def test_download_selects_usable_model_files_without_extra_formats(
    layout, fake_hub, tiny_minilm, tmp_path
):
    source = tmp_path / "repository"
    shutil.copytree(getattr(tiny_minilm, layout), source)
    for name in ["pytorch_model.bin", "onnx/model.onnx", "data/train.jsonl", "modeling.py"]:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not needed for inference")
    (source / "README.md").write_text("Model card")
    (source / "LICENSE").write_text("Model license")
    fake_hub.source = source
    cache = tmp_path / "hub-cache"
    destination = tmp_path / "downloaded"

    path = download_model(
        "org/model",
        revision="a" * 40,
        cache_dir=cache,
        local_dir=destination,
        token="test-token",
    )

    assert path == destination
    expected = {
        p.relative_to(getattr(tiny_minilm, layout))
        for p in getattr(tiny_minilm, layout).rglob("*")
        if p.is_file()
    } | {Path("README.md"), Path("LICENSE")}
    assert {p.relative_to(path) for p in path.rglob("*") if p.is_file()} == expected
    options = fake_hub.calls[0]
    assert options["repo_id"] == "org/model" and options["revision"] == "a" * 40
    assert options["cache_dir"] == cache and options["local_dir"] == destination
    assert options["token"] == "test-token" and options["local_files_only"] is False


@pytest.mark.parametrize("setting", ["1", "ON", "yes", "True"])
def test_offline_environment_forces_cached_access(setting, fake_hub, tiny_minilm, monkeypatch):
    fake_hub.destination = tiny_minilm.plain
    monkeypatch.setenv("HF_HUB_OFFLINE", setting)
    assert download_model("org/model", local_files_only=False) == tiny_minilm.plain
    assert fake_hub.calls[0]["local_files_only"] is True
    assert os.environ["HF_HUB_OFFLINE"] == setting


def test_explicit_local_only_and_anonymous_access_are_forwarded(fake_hub, tiny_minilm):
    fake_hub.destination = tiny_minilm.plain
    assert download_model("org/model", local_files_only=True, token=False) == tiny_minilm.plain
    assert fake_hub.calls[0]["local_files_only"] is True
    assert fake_hub.calls[0]["token"] is False


@pytest.mark.parametrize("variant", [False, True])
def test_incomplete_cached_snapshot_names_missing_files(variant, fake_hub, tmp_path):
    root = fake_hub.destination
    checkpoint = root / "base_model" if variant else root
    checkpoint.mkdir(parents=True)
    for name in ["config.json", "tokenizer.json"]:
        (checkpoint / name).write_text("{}")
    with pytest.raises(FileNotFoundError, match="model.safetensors") as error:
        download_model("org/model", local_files_only=True)
    assert str(checkpoint / "model.safetensors") in str(error.value)
    if variant:
        assert str(root / "adapter.safetensors") in str(error.value)


def test_incomplete_cached_variant_does_not_silently_load_without_its_adapter(
    fake_hub, tiny_minilm
):
    fake_hub.source = tiny_minilm.variant
    path = download_model("org/model")
    (path / "adapter.safetensors").unlink()
    fake_hub.source = None
    with pytest.raises(FileNotFoundError, match="adapter.safetensors"):
        download_model("org/model", local_files_only=True)


def test_mixed_download_directory_does_not_select_stale_model_files(fake_hub, tiny_minilm):
    fake_hub.source = tiny_minilm.variant
    path = download_model("org/variant")
    for source in tiny_minilm.plain.iterdir():
        shutil.copy2(source, path / source.name)
    fake_hub.source = None
    with pytest.raises(ValueError, match="clean local_dir"):
        download_model("org/plain", local_files_only=True)


def test_hub_failure_keeps_context_and_cause(fake_hub):
    class HubError(Exception):
        pass

    fake_hub.error = HubError("Access denied; check your Hugging Face login")
    with pytest.raises(RuntimeError, match="org/private.*Access denied") as error:
        download_model("org/private")
    assert error.value.__cause__ is fake_hub.error


@pytest.mark.parametrize("layout", ["plain", "variant", "retrieval"])
def test_from_pretrained_matches_local_inference(
    layout, fake_hub, tiny_minilm, require_minilm_library, tmp_path
):
    class CustomMiniLM(MiniLM):
        pass

    source = getattr(tiny_minilm, layout)
    fake_hub.source = source
    with (
        MiniLM(source, model_precision=ModelPrecision.FP32, threads=1) as local,
        CustomMiniLM.from_pretrained(
            "org/model",
            revision="v1",
            cache_dir=tmp_path / "hub-cache",
            local_files_only=True,
            token=False,
            model_precision=ModelPrecision.FP32,
            threads=1,
            library_path=require_minilm_library,
        ) as downloaded,
    ):
        assert isinstance(downloaded, CustomMiniLM)
        assert downloaded.model_precision is ModelPrecision.FP32 and downloaded.threads == 1
        assert downloaded.labels == local.labels
        assert downloaded.chunk_tokens == local.chunk_tokens
        np.testing.assert_array_equal(downloaded.embed("w4 w5 w6"), local.embed("w4 w5 w6"))
        if local.head_size:
            assert downloaded.classify("w4 w5 w6") == local.classify("w4 w5 w6")
    assert fake_hub.calls[0]["revision"] == "v1"
    assert fake_hub.calls[0]["local_files_only"] is True


def test_download_cli_does_not_load_an_engine(fake_hub, tiny_minilm, tmp_path, monkeypatch, capsys):
    fake_hub.source = tiny_minilm.variant

    def unexpected_load(*args, **kwargs):
        pytest.fail("Downloading a checkpoint must not load an inference engine")

    monkeypatch.setattr(akiki, "MiniLM", unexpected_load)
    monkeypatch.setattr(akiki, "VAD", unexpected_load)
    destination = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "akiki",
            "download",
            "org/variant",
            "--revision",
            "v1",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--local-dir",
            str(destination),
            "--local-files-only",
        ],
    )
    main()
    result = capsys.readouterr()
    assert result.out.strip() == str(destination) and result.err == ""
    assert fake_hub.calls[0]["revision"] == "v1"
    assert fake_hub.calls[0]["cache_dir"] == tmp_path / "cache"
    assert fake_hub.calls[0]["local_files_only"] is True
    assert (destination / "base_model/model.safetensors").is_file()


@pytest.mark.parametrize("layout", ["plain", "variant"])
def test_minilm_cli_runs_hub_models(
    layout, fake_hub, tiny_minilm, require_minilm_library, monkeypatch, capsys
):
    fake_hub.source = getattr(tiny_minilm, layout)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "akiki",
            "minilm",
            "--repo",
            "org/model",
            "--revision",
            "v1",
            "--local-files-only",
            "--model-precision",
            "fp32",
            "--library-path",
            str(require_minilm_library),
            "w4 w5 w6",
        ],
    )
    main()
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    with MiniLM(getattr(tiny_minilm, layout), model_precision=ModelPrecision.FP32) as model:
        if model.head_size:
            assert result["label"] == model.classify("w4 w5 w6").label
        else:
            np.testing.assert_array_equal(result["embedding"], model.embed("w4 w5 w6").tolist())
    assert fake_hub.calls[0]["revision"] == "v1"
    assert fake_hub.calls[0]["local_files_only"] is True


@pytest.mark.parametrize(
    "command", [["download", "org/private"], ["minilm", "--repo", "org/private", "hi"]]
)
def test_hub_errors_are_cli_errors(command, fake_hub, monkeypatch, capsys):
    fake_hub.error = OSError("offline cache is empty")
    monkeypatch.setattr(sys, "argv", ["akiki", *command])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"akiki {command[0]}: ")
    assert "offline cache is empty" in output.err and "Traceback" not in output.err
