"""Build the local Rust crate offline, without changing the Rust checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Shared libraries loaded by akiki.VAD and akiki.MiniLM.
PACKAGES = ("akiki-vad-engine", "akiki-minilm-engine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path,
                        default=Path(os.environ.get("AKIKI_INFERENCE_DIR", Path.home() / "projects/Akiki/akiki-inference")))
    parser.add_argument("--target-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".cache/native-target")
    args = parser.parse_args()
    manifest = args.inference_dir.expanduser().resolve() / "Cargo.toml"
    if not manifest.is_file():
        parser.error("Local akiki-inference checkout is missing; supply --inference-dir")
    cargo = shutil.which("cargo")
    if not cargo:
        parser.error("cargo is missing from PATH; use an existing local Rust toolchain")
    env = os.environ.copy()
    env["CARGO_NET_OFFLINE"] = "true"
    # Build the current host, using the baseline CPU and runtime SIMD dispatch.
    # Do not inherit target-cpu=native from a developer's shell into a release.
    for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"):
        env.pop(key, None)
    env["RUSTFLAGS"] = ""
    packages = [arg for package in PACKAGES for arg in ("--package", package)]
    subprocess.run([cargo, "build", "--offline", "--locked", "--release", "--lib", *packages,
                    "--manifest-path", str(manifest),
                    "--target-dir", str(args.target_dir.resolve()),
                    "--config", 'build.rustflags=[]'],
                   env=env, cwd=Path(__file__).resolve().parents[1], check=True)
    for package in PACKAGES:
        stem = package.replace("-", "_")
        name = (f"{stem}.dll" if sys.platform == "win32" else
                f"lib{stem}.dylib" if sys.platform == "darwin" else f"lib{stem}.so")
        library = args.target_dir.resolve() / "release" / name
        if not library.is_file():
            parser.error(f"Rust build did not produce {library}")
        print(library)


if __name__ == "__main__":
    main()
