"""Optionally bundle prebuilt C-ABI libraries in a platform-specific wheel."""

import os
import re
import shutil
import struct
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

LIBRARY_NAMES = {
    "libakiki_vad_engine.so",
    "libakiki_vad_engine.dylib",
    "akiki_vad_engine.dll",
    "libakiki_minilm_engine.so",
    "libakiki_minilm_engine.dylib",
    "akiki_minilm_engine.dll",
}


def native_libraries():
    # One path, or several separated by os.pathsep, as for PATH.
    value = os.environ.get("AKIKI_NATIVE_LIBRARY")
    if not value:
        return []
    paths = [
        Path(item).expanduser().resolve(strict=True) for item in value.split(os.pathsep) if item
    ]
    names = [path.name for path in paths]
    if not set(names) <= LIBRARY_NAMES or len(set(names)) != len(names):
        raise ValueError(
            "AKIKI_NATIVE_LIBRARY must list distinct compiled akiki-vad-engine "
            "or akiki-minilm-engine shared libraries"
        )
    return paths


def linux_platform(libraries):
    """Derive Linux wheel architecture from the actual ELF libraries, including cross-builds."""
    if not libraries or not all(path.suffix == ".so" for path in libraries):
        return None
    architectures = set()
    for path in libraries:
        with path.open("rb") as source:
            header = source.read(20)
        if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01":
            raise ValueError(f"Expected a little-endian ELF64 library: {path.name}")
        machine = struct.unpack_from("<H", header, 18)[0]
        if machine not in {62, 183}:
            raise ValueError(f"Unsupported ELF architecture in {path.name}: {machine}")
        architectures.add(machine)
    if len(architectures) != 1:
        raise ValueError("Native libraries must have the same architecture")
    machine = architectures.pop()
    return "linux_aarch64" if machine == 183 else "linux_x86_64"


def macos_platform(libraries, tag):
    """Use the bundled Mach-O architecture instead of the Python interpreter's tag."""
    if not libraries or not all(path.suffix == ".dylib" for path in libraries):
        return None
    platform = re.fullmatch(r"(macosx_\d+_\d+)_(?:x86_64|arm64|universal2)", tag)
    if platform is None:
        raise ValueError("Native macOS libraries require a macOS wheel platform")
    machines = set()
    for path in libraries:
        with path.open("rb") as source:
            header = source.read(8)
        if len(header) != 8 or header[:4] != b"\xcf\xfa\xed\xfe":
            raise ValueError(f"Expected a thin little-endian Mach-O64 library: {path.name}")
        machine = int.from_bytes(header[4:8], "little")
        if machine not in {0x01000007, 0x0100000C}:
            raise ValueError(f"Unsupported Mach-O architecture in {path.name}: {machine}")
        machines.add(machine)
    if len(machines) != 1:
        raise ValueError("Native libraries must have the same architecture")
    architecture = "arm64" if machines.pop() == 0x0100000C else "x86_64"
    return f"{platform[1]}_{architecture}"


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "akiki/native"
        # Avoid carrying a library over from a previous platform wheel build.
        if destination.exists():
            shutil.rmtree(destination)
        libraries = native_libraries()
        if libraries:
            destination.mkdir(parents=True)
        for library in libraries:
            shutil.copyfile(library, destination / library.name)


class BdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if native_libraries():
            self.root_is_pure = False

    def get_tag(self):
        python, abi, platform = super().get_tag()
        libraries = native_libraries()
        platform = linux_platform(libraries) or platform
        platform = macos_platform(libraries, platform) or platform
        # ctypes uses the C ABI; this library has no CPython ABI dependency.
        return ("py3", "none", platform) if libraries else (python, abi, platform)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return bool(native_libraries()) or super().has_ext_modules()


if __name__ == "__main__":
    setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
