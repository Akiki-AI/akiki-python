"""Native wheel architecture and cross-build selection must agree with the binaries."""

import importlib.util
import io
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
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


@pytest.mark.parametrize("machine,architecture", [(0x01000007, "x86_64"), (0x0100000C, "arm64")])
def test_macos_wheel_uses_library_architecture_with_universal_python(
    tmp_path, monkeypatch, machine, architecture
):
    setup = load_script("akiki_setup", "setup.py")
    native = tmp_path / "libakiki_vad_engine.dylib"
    native.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", machine))
    monkeypatch.setenv("AKIKI_NATIVE_LIBRARY", str(native))
    monkeypatch.setattr(
        setup.bdist_wheel,
        "get_tag",
        lambda self: ("py3", "none", "macosx_11_0_universal2"),
    )
    wheel = setup.BdistWheel(setup.BinaryDistribution())
    wheel.ensure_finalized()
    assert wheel.get_tag() == ("py3", "none", f"macosx_11_0_{architecture}")


def test_macos_wheel_rejects_mixed_library_architectures(tmp_path):
    setup = load_script("akiki_setup", "setup.py")
    vad = tmp_path / "libakiki_vad_engine.dylib"
    minilm = tmp_path / "libakiki_minilm_engine.dylib"
    vad.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x01000007))
    minilm.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x0100000C))
    with pytest.raises(ValueError, match="same architecture"):
        setup.macos_platform([vad, minilm], "macosx_11_0_universal2")


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
    from akiki import decision

    monkeypatch.setattr(decision.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(decision.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="ARM64 MiniLM support is experimental"):
        decision._load_library(tmp_path / "missing.so")


@pytest.fixture(scope="module")
def packaged_project(tmp_path_factory):
    project = tmp_path_factory.mktemp("package-contents")
    for name in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.txt",
        "README.md",
        "MANIFEST.in",
        "pyproject.toml",
        "setup.py",
        "verify_dist.py",
        "tools/build_native.py",
    ):
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name in ("src/akiki", "docs", "tests"):
        shutil.copytree(
            ROOT / name,
            project / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "native"),
        )
    for name in (
        ".akiki-lab",
        ".cache",
        "experiments",
        "local-experiments",
        "reports",
        "toy-test",
        "toy-test-typed-decisions",
        "verification",
    ):
        directory = project / name
        directory.mkdir()
        (directory / "private.jsonl").write_text('{"private": true}\n')
    (project / "playground.ipynb").write_text("local notebook")
    extra_package = project / "src/experiment_package"
    extra_package.mkdir()
    (extra_package / "__init__.py").write_text("# Local experiment\n")
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.pop("AKIKI_NATIVE_LIBRARY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_sdist, build_wheel; "
            "build_sdist('dist'); build_wheel('dist')",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return project


def verify_distributions(project, directory, *arguments):
    return subprocess.run(
        [sys.executable, str(project / "verify_dist.py"), "--dist-dir", str(directory), *arguments],
        cwd=project,
        capture_output=True,
        text=True,
    )


def test_distributions_exclude_local_work_and_pass_release_verification(packaged_project):
    project = packaged_project
    directory = project / "dist"
    with zipfile.ZipFile(next(directory.glob("*.whl"))) as wheel:
        names = wheel.namelist()
        assert "akiki/audio/vad.py" in names
        assert "akiki/decision.py" in names
        assert not any(name.startswith("akiki_training/") for name in names)
        metadata = wheel.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
        assert b"Requires-Dist: akiki-training" not in metadata
        assert not any("experiment_package" in name or "private.jsonl" in name for name in names)
    with tarfile.open(next(directory.glob("*.tar.gz"))) as source:
        names = source.getnames()
        assert any(name.endswith("/verify_dist.py") for name in names)
        assert not any("private.jsonl" in name or name.endswith(".ipynb") for name in names)
    result = verify_distributions(project, directory)
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_requires_all_desktop_wheels_when_requested(packaged_project):
    result = verify_distributions(packaged_project, packaged_project / "dist", "--require-desktop")
    assert result.returncode != 0
    assert "Windows x64 and both macOS architecture wheels are required" in result.stderr


def test_release_metadata_accepts_windows_line_endings(packaged_project, tmp_path):
    for path in (packaged_project / "dist").iterdir():
        shutil.copy2(path, tmp_path / path.name)
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in contents if name.endswith(".dist-info/METADATA"))
    contents[metadata] = contents[metadata].replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    result = verify_distributions(packaged_project, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "tag,machine,valid",
    [
        ("win_amd64", 0x8664, True),
        ("win_amd64", 0xAA64, False),
        ("macosx_11_0_x86_64", 0x01000007, True),
        ("macosx_11_0_x86_64", 0x0100000C, False),
        ("macosx_11_0_arm64", 0x0100000C, True),
        ("macosx_11_0_arm64", 0x01000007, False),
        ("macosx_11_0_universal2", 0x01000007, False),
    ],
)
def test_desktop_wheel_verification_checks_binary_architecture(
    packaged_project, tmp_path, tag, machine, valid
):
    original = next((packaged_project / "dist").glob("*.whl"))
    with zipfile.ZipFile(original) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in contents if name.endswith(".dist-info/WHEEL"))
    contents[metadata] = contents[metadata].replace(
        b"Root-Is-Purelib: true", b"Root-Is-Purelib: false"
    )
    contents[metadata] = contents[metadata].replace(b"py3-none-any", f"py3-none-{tag}".encode())
    if tag.startswith("win"):
        header = bytearray(70)
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 64)
        header[64:68] = b"PE\x00\x00"
        struct.pack_into("<H", header, 68, machine)
        names = [f"akiki_{engine}_engine.dll" for engine in ("vad", "minilm")]
    else:
        header = b"\xcf\xfa\xed\xfe" + struct.pack("<I", machine)
        names = [f"libakiki_{engine}_engine.dylib" for engine in ("vad", "minilm")]
    for name in names:
        contents["akiki/native/" + name] = bytes(header)
    wheel = tmp_path / original.name.replace("-any.whl", f"-{tag}.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    source = next((packaged_project / "dist").glob("*.tar.gz"))
    shutil.copy2(source, tmp_path / source.name)
    result = verify_distributions(
        packaged_project, tmp_path, "--wheel", str(wheel), "--require-native"
    )
    if valid:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "architecture differs from the wheel tag" in result.stderr


@pytest.mark.parametrize(
    "kind,name,message",
    [
        ("wheel", "local-experiments/private.jsonl", "Local artifact must not be distributed"),
        ("wheel", "akiki/stale_experiment.py", "Unexpected/missing package files"),
        ("sdist", "local-experiments/private.jsonl", "Local artifact must not be distributed"),
        ("sdist", "src/akiki/stale_experiment.py", "Unexpected/missing package files"),
        ("wheel", "akiki_training/__init__.py", "Private training artifact"),
        ("sdist", "src/akiki_training/__init__.py", "Private training artifact"),
        ("sdist", "akiki-training/src/backend.py", "Private training artifact"),
        (
            "sdist",
            "private/akiki_training-0.1.0-py3-none-any.whl",
            "Private training artifact",
        ),
        ("sdist", "private/akiki-training-0.1.0.tar.gz", "Private training artifact"),
    ],
)
def test_release_verification_rejects_local_or_stale_files(
    packaged_project, tmp_path, kind, name, message
):
    for path in (packaged_project / "dist").iterdir():
        shutil.copy2(path, tmp_path / path.name)
    if kind == "wheel":
        with zipfile.ZipFile(next(tmp_path.glob("*.whl")), "a") as archive:
            archive.writestr(name, "local experiment")
    else:
        source = next(tmp_path.glob("*.tar.gz"))
        with tarfile.open(source) as archive:
            members = [
                (member, archive.extractfile(member).read() if member.isfile() else None)
                for member in archive.getmembers()
            ]
        prefix = source.name.removesuffix(".tar.gz")
        data = b"local experiment"
        extra = tarfile.TarInfo(f"{prefix}/{name}")
        extra.size = len(data)
        members.append((extra, data))
        with tarfile.open(source, "w:gz") as archive:
            for member, data in members:
                archive.addfile(member, io.BytesIO(data) if data is not None else None)
    result = verify_distributions(packaged_project, tmp_path)
    assert result.returncode != 0
    assert message in result.stderr, result.stdout + result.stderr
