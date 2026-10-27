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
native="$PWD/.cache/native-target/release"
AKIKI_NATIVE_LIBRARY="$native/libakiki_vad_engine.so:$native/libakiki_minilm_engine.so" \
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
The native wheel adds the compiled Rust VAD and MiniLM libraries and is
preferred by pip and uv on compatible hosts. `AKIKI_NATIVE_LIBRARY` lists the
libraries to bundle, separated by `:` (`;` on Windows); `verify_dist.py`
requires both in every native wheel.

MiniLM weights are never bundled. Users pass a local model path to
`akiki.MiniLM`. The test suite writes tiny random checkpoints and adapters to
test the binding without real weights. For the real-model tests, set
`AKIKI_MINILM_MODEL` to a local checkpoint or Ariadne variant folder. A Clover
release's recorded test predictions are checked automatically. Set
`AKIKI_MINILM_REFERENCES` to PyTorch reference JSONL for that model, in the
format of the engine's local-model tests, to check embeddings and logits. Other hosts must install an optional runtime using
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
`fp32`, and that `akiki.minilm._load_library()` loads the installed MiniLM
library. When no precision is specified, optional ONNX and PyTorch engines use
FP32; an explicit non-FP32 request must require Rust.

## Publish from GitHub Actions

The workflow is `.github/workflows/publish.yml`. It builds and verifies the
portable wheel, source archive, and Linux x86_64 native wheel. It tests the
native loader, Python API, and clean installations before saving the artifacts.
Models come exclusively from `src/akiki/models/`; no model is downloaded.

Pushes to `main` and manual runs from the Actions tab build and test without
publishing. Publishing a GitHub release runs the same checks and then uploads
the verified artifacts to PyPI using Trusted Publishing. The release tag must
be `v` followed by the version in `pyproject.toml`, for example `v0.1.0`. Update
all the version locations listed above first, and use a version that has not
already been uploaded to PyPI.

The existing PyPI `akiki` project must trust owner `Akiki-AI`, repository
`akiki-python`, workflow filename `publish.yml`, with the environment left
blank. Only the publishing job has `id-token: write`; no PyPI token is stored.

The native source is pinned by `AKIKI_INFERENCE_REV` in the workflow. Update
that commit deliberately when changing the engine. It must be reachable on
GitHub, preferably on `main`, and its MiniLM C ABI (`src/ffi.rs` in
`akiki-minilm-engine`) must match `src/akiki/minilm.py`. Both repositories remain
private. Source checkout uses a dedicated read-only SSH deploy key:

1. Enable deploy keys in the organization's settings if they are disabled.
2. Add the public key under `akiki-inference` Settings > Deploy keys, leaving
   Allow write access unchecked.
3. Store the matching private key as the `AKIKI_INFERENCE_SSH_KEY` Actions
   secret in `akiki-python`.

The private key must never be committed. It is used only for source checkout
and is separate from PyPI Trusted Publishing. To rotate it, install a new
public key and update the secret before deleting the old deploy key.

The workflow targets glibc 2.34 and newer. Auditwheel must verify that floor;
a build that needs newer libc symbols fails rather than publishing a renamed
wheel. The engine's self-contained safetensors and configuration tests run in
CI; legacy Rust tests that require external `.akv` fixtures are not part of
this workflow. The Python tests use the exact bundled release models.

## Publish from a local shell

Set `UV_PUBLISH_TOKEN` in your own shell or use configured trusted publishing.
Do not store credentials in this repository.

```bash
./publish.sh --dry-run
./publish.sh
```

The script verifies and uploads all wheels for the current version plus its
source archive. It refuses a release with no native wheel or stale description.
It does not upload artifacts from previous versions.
