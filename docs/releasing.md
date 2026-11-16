# Releasing Akiki

The public PyPI description comes from `README.md`. Keep it focused on package
features and usage. Build and publishing instructions belong here.

## Branches and release preparation

Use `rc-0.2.0` for integration and release checks. Keep `main` unchanged until
the code has passed human review.
When the RC is ready, create `release-0.2.0` from the tested RC commit.
Set the release version on that branch before you build the release artifacts.
Creating or pushing these branches does not publish a package to PyPI.
The branch name does not set the package version. Read the current value with
`python3 verify_dist.py --version`.

Update the version in `pyproject.toml`, `src/akiki/__init__.py`, the version
assertion in `tests/test_vad.py`, and the package version in
`src/akiki/models/release.json` and `src/akiki/models/MODEL_CARD.md`. Preserve the model identity
and weight checksums unless deliberately releasing a different model.

PyPI takes a release's description from its first upload. A description change
requires a new version; an additional wheel does not update an existing page.
Previously uploaded filenames cannot be replaced. Keep previous artifacts intact.

## Repository and package contents

Commit package source, public documentation, synthetic tests, and release tools.
Keep datasets, notebooks, teacher responses, trained specialist adapters, and
experiment outputs local. Use `local-experiments/` or a separate workspace for
new experiments. Existing toy-model and verification directories are ignored.
Preserve the released VAD assets in `src/akiki/models/`.

Stage files by name. Before a commit, inspect the staged files:

```bash
git diff --cached --stat
git diff --cached --check
git status --short
```

Build from a clean checkout of the tested commit. The wheel contains the
`akiki` package, released VAD assets, and optional native libraries.
The source archive also contains the public build instructions, synthetic
tests, and release verification tools. Packaging rules exclude local experiment
directories and notebooks. `verify_dist.py` rejects local artifacts and
unexpected package files in the built archives.

The training implementation, dataset tools, and Lab belong to separate
companion packages. Akiki's `training` extra selects
`akiki-training[ariadne]`. Before publication, make the tested companion version
available from the intended package index. Test the extra in a clean environment
without an editable companion checkout. Check that base inference still works
without training dependencies. Install `akiki-training-lab` separately for
`akiki-lab --open`.

## Build the Linux x86_64 release

Use an existing local Rust toolchain, Cargo cache and `akiki-inference` checkout.
The x86_64 wheel includes VAD and MiniLM. Starting with 0.1.4, the ARM64 wheel
also includes both engines. ARM64 MiniLM support is experimental. The existing
pinned Rust revision is used for both architectures.

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
requires both on x86_64 and ARM64. Linux wheel architecture is
derived from the ELF libraries, so cross-built ARM64 libraries cannot be
accidentally labelled as x86_64.

## Build the Linux ARM64 VAD and experimental MiniLM wheel

On a native ARM64 Linux builder with cached tools and Cargo dependencies:

```bash
python3 tools/build_native.py --inference-dir "$AKIKI_INFERENCE_DIR"
native="$PWD/.cache/native-target/release"
AKIKI_NATIVE_LIBRARY="$native/libakiki_vad_engine.so:$native/libakiki_minilm_engine.so" \
    uv build --offline --wheel --out-dir .cache/raw-wheels
uv tool run --offline --with patchelf auditwheel repair \
    --plat manylinux_2_28_aarch64 --wheel-dir dist .cache/raw-wheels/*aarch64.whl
python3 verify_dist.py --all-wheels --require-native --require-arm64
```

Build the source archive alongside the wheels before verification. The native
build helper selects both VAD and MiniLM on ARM64. For cross-compilation, it also accepts
`--target aarch64-unknown-linux-gnu.2.28 --zig` with an existing ARM64 Rust std,
`cargo-zigbuild`, and Zig installation. Cross-build output is under
`.cache/native-target/aarch64-unknown-linux-gnu/release/`.

ARM64 INT8 projections use the pinned KleidiAI kernel subset through the shared
Rust layer/kernel crates. Release builds keep the baseline ARM64 target and
select Neon DotProd at runtime; older CPUs and other precisions retain portable
execution. The VAD backend was measured and checked bitwise on a physical Pi 5. ARM64
MiniLM is experimental; emulation checks establish correctness, not hardware
performance.
CI also runs shared kernel/layer tests and quantized VAD streaming tests on its
native ARM64 builder. Include THIRD_PARTY_NOTICES.txt in every artifact and
preserve its Arm copyright and Apache-2.0 terms when updating the pinned engine.
The package remains Alpha; MiniLM is supported in the Linux x86_64 and ARM64
wheels, with ARM64 MiniLM support considered experimental.

MiniLM weights are never bundled. Users pass a local model path to
`akiki.MiniLM`, or explicitly fetch a compatible checkpoint with
`MiniLM.from_pretrained()` or `akiki download` using the optional
`akiki[huggingface]` extra. Model loading during release builds and tests must
remain offline. CI can fetch pinned source and build dependencies.
Hub tests use a fake download boundary and locally generated checkpoints.
The test suite writes tiny random checkpoints and adapters to test the binding
without real weights. For the real-model tests, set
`AKIKI_MINILM_MODEL` to a local checkpoint or Ariadne variant folder. A Clover
release's recorded test predictions are checked automatically. Set
`AKIKI_MINILM_REFERENCES` to PyTorch reference JSONL for that model, in the
format of the engine's local-model tests, to check embeddings and logits. Other hosts must install an optional runtime using
`akiki[onnx]` or `akiki[pytorch]`. The base package requires neither runtime;
verify that the extras remain independent in every release artifact.
Run the training tests with the tested companion installed as well.
Read the skipped-test summary; native release jobs do not install the training
companion. Their default test run can skip the optional training checks.

The Linux platform floor above was verified with auditwheel for the current
library. It requires x86_64 and glibc 2.34 or newer. Auditwheel must succeed;
renaming a wheel cannot make its binary compatible with an older libc. For
broader Linux compatibility, rebuild using an older manylinux toolchain and
validate with auditwheel. Build and test macOS and Windows libraries on their
target platforms before adding those wheels to `dist/`.

Validate the Markdown and metadata with `twine check` for every new artifact.
Test a clean install with both portable and platform wheels available and
confirm that `VAD().engine is Engine.RUST`, loading the installed library
(`Engine` is exported from `akiki`).
Confirm that `VAD().model_precision is ModelPrecision.INT8` and
`VAD().fft_precision is FFTPrecision.FP32` (both enums are exported from `akiki.vad`),
and that `akiki.decision._load_library()` loads the installed MiniLM
library. When no precision is specified, optional ONNX and PyTorch engines use
FP32; an explicit non-FP32 request must require Rust.

## Publish from GitHub Actions

The workflow is `.github/workflows/publish.yml`. It builds and verifies the
portable wheel, source archive, Linux x86_64 native wheel, and experimental
Linux ARM64 VAD and experimental MiniLM wheel. ARM64 builds in manylinux_2_28 on a native ARM64 runner;
x86_64 retains the glibc 2.34 floor. Both jobs must pass before publishing. It tests the
native loader, Python API, and clean installations before saving the artifacts.
Models come exclusively from `src/akiki/models/`; no model is downloaded.

Run the workflow manually on the RC or release branch to build and test without
publishing. Leave the `publish` input disabled. Pushes to `main` also run these
checks after human review; do not update `main` to start an RC build.
Publishing a GitHub release runs the same checks and then uploads
the verified artifacts to PyPI using Trusted Publishing. The release tag must
be `v` followed by the version in `pyproject.toml`, for example `v0.1.0`. Update
all the version locations listed above first, and use a version that has not
already been uploaded to PyPI.

If publication fails before uploading, fix the workflow on the release branch
and run it manually with the `publish` input enabled. The retry requires an
existing GitHub release and checks that package source matches its version tag;
only the workflow and this release document may differ. This preserves the tag
while allowing CI-only repairs. Both native build jobs run again before upload.

The existing PyPI `akiki` project must trust owner `Akiki-AI`, repository
`akiki-python`, workflow filename `publish.yml`, with the environment left
blank. Only the publishing job has `id-token: write`; no PyPI token is stored.

The native source is pinned by `AKIKI_INFERENCE_REV` in the workflow. Update
that commit deliberately when changing the engine. It must be reachable on
GitHub and have passed the required engine review. Its MiniLM C ABI
(`crates/akiki-minilm-engine/src/ffi.rs`) must match `src/akiki/decision.py`. Both repositories remain
private. Source checkout uses a dedicated read-only SSH deploy key:

1. Enable deploy keys in the organization's settings if they are disabled.
2. Add the public key under `akiki-inference` Settings > Deploy keys, leaving
   Allow write access unchecked.
3. Store the matching private key as the `AKIKI_INFERENCE_SSH_KEY` Actions
   secret in `akiki-python`.

The private key must never be committed. It is used only for source checkout
and is separate from PyPI Trusted Publishing. To rotate it, install a new
public key and update the secret before deleting the old deploy key.

The x86_64 workflow targets glibc 2.34 and ARM64 targets glibc 2.28. Auditwheel must verify those floors;
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
