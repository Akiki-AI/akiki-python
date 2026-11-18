"""Load native engines from the package or an explicit local path."""

from pathlib import Path

import pytest

from akiki import _engines, decision
from akiki.audio.vad import VAD, Engine, ModelPrecision


@pytest.fixture(params=["win32", "darwin", "linux"])
def native_library(request, tmp_path, monkeypatch):
    monkeypatch.setattr(_engines.sys, "platform", request.param)
    monkeypatch.setattr(_engines, "__file__", str(tmp_path / "site-packages/akiki/_engines.py"))
    for variable in ("AKIKI_VAD_LIBRARY", "AKIKI_MINILM_LIBRARY"):
        monkeypatch.delenv(variable, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("AKIKI_INFERENCE_DIR", str(tmp_path / "inference"))
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "cargo"))
    monkeypatch.setenv("CARGO_BUILD_TARGET", "target-triple")
    monkeypatch.chdir(tmp_path)
    return (tmp_path / "site-packages/akiki/native").resolve()


@pytest.fixture(
    params=[
        ("akiki_vad_engine", "AKIKI_VAD_LIBRARY"),
        ("akiki_minilm_engine", "AKIKI_MINILM_LIBRARY"),
    ]
)
def engine_library(request, native_library, tmp_path):
    stem, variable = request.param
    name = _engines.library_name(stem)
    # Existing developer outputs must not become runtime candidates.
    for directory in (
        tmp_path,
        tmp_path / ".cache/native-target/release",
        tmp_path / "home/projects/Akiki/akiki-inference/target/target-triple/release",
        tmp_path / "inference/target/target-triple/release",
        tmp_path / "cargo/target-triple/release",
        tmp_path / "cargo/target-triple/debug",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"developer library")
    return stem, variable, native_library / name


def test_automatic_search_checks_only_the_package(engine_library):
    stem, variable, bundled = engine_library
    assert list(_engines.library_candidates(stem=stem, variable=variable)) == [bundled]


def test_explicit_path_precedes_environment_and_package(engine_library, tmp_path, monkeypatch):
    stem, variable, bundled = engine_library
    environment = tmp_path / "environment library" / bundled.name
    monkeypatch.setenv(variable, str(environment))
    assert list(_engines.library_candidates(stem=stem, variable=variable)) == [environment]

    explicit = tmp_path / "explicit library" / bundled.name
    for path in (explicit, str(explicit)):
        assert list(_engines.library_candidates(path, stem, variable)) == [explicit]

    home_library = Path.home() / "local library" / bundled.name
    relative = Path("~/local library") / bundled.name
    assert list(_engines.library_candidates(relative, stem, variable)) == [home_library]
    monkeypatch.setenv(variable, str(relative))
    assert list(_engines.library_candidates(stem=stem, variable=variable)) == [home_library]


def load_engine(stem, library_path=None):
    if stem == "akiki_vad_engine":
        return VAD(
            engine=Engine.AKIKI,
            model_precision=ModelPrecision.FP32,
            library_path=library_path,
        )
    return decision._load_library(library_path)


@pytest.mark.parametrize("present", [False, True])
def test_consumers_do_not_load_developer_libraries(engine_library, monkeypatch, present):
    stem, variable, bundled = engine_library
    if present:
        bundled.parent.mkdir(parents=True)
        bundled.write_bytes(b"incompatible bundled library")
    attempts = []

    def load(path):
        attempts.append(Path(path))
        raise OSError("incompatible library")

    monkeypatch.setattr(_engines.ctypes, "CDLL", load)
    with pytest.raises(RuntimeError, match="No compatible Akiki"):
        load_engine(stem)
    assert attempts == ([bundled] if present else [])


@pytest.mark.parametrize("present", [False, True])
def test_failed_override_does_not_load_another_library(
    engine_library, tmp_path, monkeypatch, present
):
    stem, variable, bundled = engine_library
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled library")
    explicit = tmp_path / "custom" / bundled.name
    if present:
        explicit.parent.mkdir()
        explicit.write_bytes(b"incompatible explicit library")
    attempts = []

    def load(path):
        attempts.append(Path(path))
        raise OSError("incompatible library")

    monkeypatch.setattr(_engines.ctypes, "CDLL", load)
    for library_path in (explicit, None):
        attempts.clear()
        monkeypatch.setenv(variable, str(explicit if library_path is None else bundled))
        with pytest.raises(RuntimeError, match=variable):
            load_engine(stem, library_path)
        assert attempts == ([explicit.resolve()] if present else [])
