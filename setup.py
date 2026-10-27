"""Optionally bundle prebuilt C-ABI libraries in a platform-specific wheel."""
import os
from pathlib import Path
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


LIBRARY_NAMES = {
    "libakiki_vad_engine.so", "libakiki_vad_engine.dylib", "akiki_vad_engine.dll",
    "libakiki_minilm_engine.so", "libakiki_minilm_engine.dylib", "akiki_minilm_engine.dll",
}


def native_libraries():
    # One path, or several separated by os.pathsep, as for PATH.
    value = os.environ.get("AKIKI_NATIVE_LIBRARY")
    if not value:
        return []
    paths = [Path(item).expanduser().resolve(strict=True) for item in value.split(os.pathsep) if item]
    names = [path.name for path in paths]
    if not set(names) <= LIBRARY_NAMES or len(set(names)) != len(names):
        raise ValueError("AKIKI_NATIVE_LIBRARY must list distinct compiled akiki-vad-engine "
                         "or akiki-minilm-engine shared libraries")
    return paths


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
        # ctypes uses the C ABI; this library has no CPython ABI dependency.
        return ("py3", "none", platform) if native_libraries() else (python, abi, platform)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return bool(native_libraries()) or super().has_ext_modules()


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
