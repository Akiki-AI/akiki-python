"""Optionally bundle a prebuilt C-ABI library in a platform-specific wheel."""
import os
from pathlib import Path
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


def native_library():
    value = os.environ.get("AKIKI_NATIVE_LIBRARY")
    if not value:
        return None
    path = Path(value).expanduser().resolve(strict=True)
    if path.name not in {"libakiki_vad_engine.so", "libakiki_vad_engine.dylib", "akiki_vad_engine.dll"}:
        raise ValueError("AKIKI_NATIVE_LIBRARY must name a compiled akiki-vad-engine shared library")
    return path


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "akiki/native"
        # Avoid carrying a library over from a previous platform wheel build.
        if destination.exists():
            shutil.rmtree(destination)
        library = native_library()
        if library:
            destination.mkdir(parents=True)
            shutil.copyfile(library, destination / library.name)


class BdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if native_library():
            self.root_is_pure = False

    def get_tag(self):
        python, abi, platform = super().get_tag()
        # ctypes uses the C ABI; this library has no CPython ABI dependency.
        return ("py3", "none", platform) if native_library() else (python, abi, platform)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return bool(native_library()) or super().has_ext_modules()


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
