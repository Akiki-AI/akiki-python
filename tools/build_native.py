"""Build the local Rust crate offline, without changing the Rust checkout."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Shared libraries loaded by akiki.VAD and akiki.MiniLM.
PACKAGES = ("akiki-vad-engine", "akiki-minilm-engine")


def packages_for_target(target=None):
    architecture = target.split("-", 1)[0] if target else platform.machine().lower()
    return PACKAGES[:1] if architecture in {"aarch64", "arm64"} else PACKAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path(
            os.environ.get("AKIKI_INFERENCE_DIR", Path.home() / "projects/Akiki/akiki-inference")
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".cache/native-target",
    )
    parser.add_argument(
        "--target", help="Optional Rust target triple; ARM64 builds include VAD only"
    )
    parser.add_argument(
        "--zig", action="store_true", help="Use cargo-zigbuild for a portable Linux cross-build"
    )
    args = parser.parse_args()
    manifest = args.inference_dir.expanduser().resolve() / "Cargo.toml"
    if not manifest.is_file():
        parser.error("Local akiki-inference checkout is missing; supply --inference-dir")
    cargo = shutil.which("cargo")
    if not cargo:
        parser.error("cargo is missing from PATH; use an existing local Rust toolchain")
    if args.zig and (not args.target or not shutil.which("cargo-zigbuild")):
        parser.error("--zig requires --target and an existing cargo-zigbuild installation")
    env = os.environ.copy()
    env["CARGO_NET_OFFLINE"] = "true"
    # Use the baseline CPU; ISA-specific kernels are selected at runtime.
    # Do not inherit target-cpu=native from a developer's shell into a release.
    for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"):
        env.pop(key, None)
    env["RUSTFLAGS"] = ""
    selected = packages_for_target(args.target)
    packages = [arg for package in selected for arg in ("--package", package)]
    target = ["--target", args.target] if args.target else []
    subprocess.run(
        [
            cargo,
            "zigbuild" if args.zig else "build",
            "--offline",
            "--locked",
            "--release",
            "--lib",
            *packages,
            *target,
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(args.target_dir.resolve()),
            "--config",
            "build.rustflags=[]",
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    output = args.target_dir.resolve()
    if args.target:
        output /= args.target.split(".", 1)[0]
    system = "linux" if args.target and "linux" in args.target else sys.platform
    for package in selected:
        stem = package.replace("-", "_")
        name = (
            f"{stem}.dll"
            if system == "win32"
            else f"lib{stem}.dylib"
            if system == "darwin"
            else f"lib{stem}.so"
        )
        library = output / "release" / name
        if not library.is_file():
            parser.error(f"Rust build did not produce {library}")
        print(library)


if __name__ == "__main__":
    main()
