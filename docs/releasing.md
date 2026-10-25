# Releasing Akiki

The public PyPI description comes from `README.md`. Keep it focused on package
features and usage. Build and publishing instructions belong here.

Update the version in `pyproject.toml`, `src/akiki/__init__.py`, the version
assertion in `tests/test_vad.py`, and the package version in
`src/akiki/models/release.json` and `MODEL_CARD.md`. Preserve the model identity
and weight checksums unless deliberately releasing a different model.

PyPI takes a release's description from its first upload. A description change
requires a new version; an additional wheel does not update an existing page.
Previously uploaded filenames cannot be replaced. Keep previous artifacts intact.

## Build the Linux x86_64 release

Use an existing local Rust toolchain, Cargo cache and `akiki-inference` checkout.
The engine currently supports x86_64. Do not advertise ARM64 Rust wheels until
the engine supports that architecture and those builds have been tested.

```bash
python3 tools/build_native.py --inference-dir "$AKIKI_INFERENCE_DIR"
uv build --offline
AKIKI_NATIVE_LIBRARY="$PWD/.cache/native-target/release/libakiki_vad_engine.so" \
    uv build --offline --wheel --out-dir .cache/raw-wheels
version=$(python3 verify_dist.py --version)
uv tool run --offline --with patchelf auditwheel repair \
    --plat manylinux_2_34_x86_64 --wheel-dir dist \
    ".cache/raw-wheels/akiki-${version}-py3-none-linux_x86_64.whl"
python3 verify_dist.py --all-wheels --require-native
```

These commands use cached build tools and dependencies. Models must come from
the existing local package assets; never download model artifacts as part of
a build. The portable wheel and source archive include the same model assets.
The native wheel adds the compiled Rust library and is preferred by pip and uv
on compatible hosts. Other hosts must install an optional runtime using
`akiki[onnx]` or `akiki[pytorch]`. The base package requires neither runtime;
verify that the extras remain independent in every release artifact.

The Linux platform floor above was verified with auditwheel for the current
library. It requires x86_64 and glibc 2.34 or newer. Auditwheel must succeed;
renaming a wheel cannot make its binary compatible with an older libc. For
broader Linux compatibility, rebuild using an older manylinux toolchain and
validate with auditwheel. Build and test macOS and Windows libraries on their
target platforms before adding those wheels to `dist/`.

Validate the Markdown and metadata with `twine check` for every new artifact.
Test a clean install with both portable and platform wheels available and
confirm that `VAD().engine` reports `rust`, loading the installed library.
Confirm that `VAD().model_precision` is `int8` and `VAD().fft_precision` is
`fp32`. When no precision is specified, optional ONNX and PyTorch engines use
FP32; an explicit non-FP32 request must require Rust.

## Publish

Set `UV_PUBLISH_TOKEN` in your own shell or use configured trusted publishing.
Do not store credentials in this repository.

```bash
./publish.sh --dry-run
./publish.sh
```

The script verifies and uploads all wheels for the current version plus its
source archive. It refuses a release with no native wheel or stale description.
It does not upload artifacts from previous versions.
