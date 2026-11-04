"""Native wheel architecture and cross-build selection must agree with the binaries."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library(path, machine):
    header = bytearray(20)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, machine)
    path.write_bytes(header)
    return path


@pytest.mark.parametrize("machine,tag", [(62, "linux_x86_64"), (183, "linux_aarch64")])
def test_wheel_platform_follows_elf_instead_of_build_host(tmp_path, monkeypatch, machine, tag):
    setup = load_script("akiki_setup", "setup.py")
    native = library(tmp_path / "libakiki_vad_engine.so", machine)
    monkeypatch.setenv("AKIKI_NATIVE_LIBRARY", str(native))
    wheel = setup.BdistWheel(setup.BinaryDistribution())
    wheel.ensure_finalized()
    assert not wheel.root_is_pure
    assert wheel.get_tag() == ("py3", "none", tag)


def test_mixed_architectures_are_rejected_and_arm64_minilm_is_allowed(tmp_path):
    setup = load_script("akiki_setup", "setup.py")
    vad = library(tmp_path / "libakiki_vad_engine.so", 183)
    minilm = library(tmp_path / "libakiki_minilm_engine.so", 62)
    with pytest.raises(ValueError, match="same architecture"):
        setup.linux_platform([vad, minilm])
    library(minilm, 183)
    assert setup.linux_platform([vad, minilm]) == "linux_aarch64"


def test_arm64_cross_build_selects_both_engines_and_the_target_output(
    tmp_path, monkeypatch, capsys
):
    builder = load_script("akiki_build_native", "tools/build_native.py")
    (tmp_path / "Cargo.toml").write_text("[workspace]\n")
    output = tmp_path / "out"
    target = "aarch64-unknown-linux-gnu.2.28"
    expected = [
        output / "aarch64-unknown-linux-gnu/release" / f"libakiki_{engine}_engine.so"
        for engine in ("vad", "minilm")
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_native.py",
            "--inference-dir",
            str(tmp_path),
            "--target-dir",
            str(output),
            "--target",
            target,
            "--zig",
        ],
    )
    monkeypatch.setattr(builder.shutil, "which", lambda command: "/local/bin/" + command)
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    monkeypatch.setenv("CARGO_BUILD_TARGET", "x86_64-unknown-linux-gnu")

    def build(command, *, env, cwd, check):
        assert command[1] == "zigbuild" and "--offline" in command and "--locked" in command
        assert command[command.index("--target") + 1] == target
        assert command[command.index("--package") + 1] == "akiki-vad-engine"
        assert "akiki-minilm-engine" in command
        assert env["RUSTFLAGS"] == "" and "CARGO_BUILD_TARGET" not in env
        expected[0].parent.mkdir(parents=True)
        for path in expected:
            path.touch()

    monkeypatch.setattr(builder.subprocess, "run", build)
    builder.main()
    assert capsys.readouterr().out.splitlines() == [str(path) for path in expected]


def test_arm64_minilm_error_explains_experimental_support(tmp_path, monkeypatch):
    import akiki.minilm as minilm

    monkeypatch.setattr(minilm.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(minilm.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="ARM64 MiniLM support is experimental"):
        minilm._load_library(tmp_path / "missing.so")
