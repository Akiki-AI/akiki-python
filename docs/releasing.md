# Releasing Akiki

The public PyPI description comes from `README.md`. Keep it focused on package
features and usage. Build and publishing instructions belong here.

Akiki provides VAD and decision models.
The primary classes are `akiki.audio.vad.VAD` and `akiki.decision.DecisionModel`.
Current native decision inference supports MiniLM/BERT exports.
The library filenames below retain the name of that implementation.

## Branches and release preparation

Use `main` for reviewed source and documentation.
Create each change on a short-lived branch from `main`.
Use a descriptive `feat/`, `fix/`, `docs/`, or `chore/` name.
Open a pull request with `main` as its base.
Require the `lint` and `tests` checks before merging.
Resolve review conversations before merging.
Delete the branch after its pull request is merged.

The public CI workflow runs lint and tests on pushes and pull requests.
Its tests use public dependencies and synthetic local checkpoints.
Private engine and training checks run on `main` and trusted manual builds.
Fork PR jobs do not receive private checkout credentials.

Use version tags to record releases and release candidates.
For example, use `v0.2.1` for version `0.2.1` and `v0.2.2rc1` for `0.2.2rc1`.
Do not create an RC branch or release branch for each version.
Keep published tags unchanged.
Create a maintenance branch only when an older version needs separate fixes.

Set the next package version in a pull request before its release.
Read the current value with `python3 verify_dist.py --version`.
Merge the version change and wait for the native builds and private training checks.
Tag the tested `main` commit with `v` followed by that exact version.
Creating a branch or pushing a tag does not publish a package to PyPI.
Publishing the GitHub release starts PyPI publication.

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

The training implementation, dataset tools, and Lab remain private.
Do not publish or bundle their code with Akiki.
The public package retains the training API wrapper.
Its `training` extra installs runtime dependencies only.
Authorized users must install the private companion from a local checkout or wheel.

Build the private companion from its pinned source revision for each release check.
Test the public wrapper with that separate companion wheel.
Check that base inference and server teacher imports work without that companion.
Verify that public wheel metadata has no dependency on the unpublished companion.
Verify that public archives contain no `akiki_training` package.
Keep the private wheel and its validation outputs under `verification/`.

## Build the Linux x86_64 release

Use an existing local Rust toolchain, Cargo cache and `akiki-inference` checkout.
The x86_64 wheel includes the VAD and decision-model runtimes.
The current decision runtime supports MiniLM/BERT exports.
Starting with 0.1.4, the ARM64 wheel also includes both native libraries.
ARM64 MiniLM/BERT inference is experimental.
Use the existing pinned engine revision for both architectures.

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
The native wheel adds the compiled Akiki VAD and decision-model libraries.
The current decision library is named `libakiki_minilm_engine.so` on Linux.
Pip and uv prefer the native wheel on compatible hosts. `AKIKI_NATIVE_LIBRARY` lists the
libraries to bundle, separated by `:` (`;` on Windows); `verify_dist.py`
requires both on x86_64 and ARM64. Linux wheel architecture is
derived from the ELF libraries, so cross-built ARM64 libraries cannot be
accidentally labelled as x86_64.

The build helper does not install the libraries into the Python package.
To test a source checkout with these Linux builds, set both runtime paths:

```bash
export AKIKI_VAD_LIBRARY="$native/libakiki_vad_engine.so"
export AKIKI_MINILM_LIBRARY="$native/libakiki_minilm_engine.so"
```

Runtime discovery checks only the package's `native` directory when no path is set.
It does not use `AKIKI_INFERENCE_DIR`, Cargo variables, or developer build directories.
Use the library filenames for your target system with Windows or macOS builds.

## Build the Linux ARM64 wheel

The wheel includes VAD and the current MiniLM/BERT decision runtime.
ARM64 support for this decision runtime is experimental.
Use a native ARM64 Linux builder with cached tools and Cargo dependencies:

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
build helper selects VAD and the current decision runtime on ARM64.
For cross-compilation, it also accepts
`--target aarch64-unknown-linux-gnu.2.28 --zig` with an existing ARM64 Rust std,
`cargo-zigbuild`, and Zig installation. Cross-build output is under
`.cache/native-target/aarch64-unknown-linux-gnu/release/`.

ARM64 INT8 projections use the pinned KleidiAI kernel subset through the shared
native layer/kernel crates. Release builds keep the baseline ARM64 target and
select Neon DotProd at runtime; older CPUs and other precisions retain portable
execution. The VAD backend was measured and checked bitwise on a physical Pi 5. ARM64
MiniLM is experimental; emulation checks establish correctness, not hardware
performance.
CI also runs shared kernel/layer tests and quantized VAD streaming tests on its
native ARM64 builder. Include THIRD_PARTY_NOTICES.txt in every artifact and
preserve its Arm copyright and Apache-2.0 terms when updating the pinned engine.
The package remains Alpha.
The Linux x86_64 and ARM64 wheels support MiniLM/BERT decision-model exports.
ARM64 support for this decision runtime remains experimental.

Decision-model weights are never bundled.
Users select a local base checkpoint through `DecisionModel.base_model`.
The lower-level `akiki.decision.MiniLM` class gives direct access to this family.
Users can explicitly fetch a compatible checkpoint with
`MiniLM.from_pretrained()` or `akiki download` using the optional
`akiki[huggingface]` extra. Model loading during release builds and tests must
remain offline. CI can fetch pinned source and build dependencies.
Hub tests use a fake download boundary and locally generated checkpoints.
The test suite writes tiny random checkpoints and adapters to test the binding
without real weights. For the real-model tests, set
`AKIKI_MINILM_MODEL` to a local checkpoint or Ariadne variant folder. A Clover
release's recorded test predictions are checked automatically. Set
`AKIKI_MINILM_REFERENCES` to PyTorch reference JSONL for that model, in the
format of the engine's local-model tests, to check embeddings and logits.
For VAD compatibility without the Akiki engine, install `akiki[onnx]` or `akiki[pytorch]`.
Current decision-model inference requires the Akiki engine.
The base package requires neither compatibility runtime.
Verify that the extras remain independent in every release artifact.
Run the training tests with the tested companion installed as well.
Read the skipped-test summary. Native build jobs do not install the training
companion. The separate training job installs the companion and its runtimes.
It runs the training checks against the installed x86_64 wheel.

The Linux platform floor above was verified with auditwheel for the current
library. It requires x86_64 and glibc 2.34 or newer. Auditwheel must succeed;
renaming a wheel cannot make its binary compatible with an older libc. For
broader Linux compatibility, rebuild using an older manylinux toolchain and
validate with auditwheel. Build and test macOS and Windows libraries on their
target platforms before adding those wheels to `dist/`.

Validate the Markdown and metadata with `twine check` for every new artifact.
Test a clean install with both portable and platform wheels available and
confirm that `VAD().engine is Engine.AKIKI`, loading the installed library
(`Engine` is exported from `akiki.audio.vad`).
Confirm that `VAD().model_precision is ModelPrecision.INT8` and
`VAD().fft_precision is FFTPrecision.FP32` (both enums are exported from `akiki.audio.vad`),
and that `akiki.decision._load_library()` loads the installed MiniLM
library. When no precision is specified, ONNX and PyTorch compatibility engines use
FP32; an explicit non-FP32 request must require the Akiki engine.

## Build Windows and macOS wheels

The release workflow builds Windows x64, macOS Intel, and macOS Apple Silicon wheels.
It uses Windows and macOS GitHub-hosted runners.
No local Windows or Mac computer is required.
Each wheel contains the VAD and decision-model libraries.
The jobs use the pinned engine revision and baseline CPU settings.

Windows wheels use the `win_amd64` tag.
Mac wheels use separate `macosx_11_0_x86_64` and `macosx_11_0_arm64` tags.
The macOS deployment target is 11.0. CI tests run on macOS 15.
Windows ARM64 is not part of this build matrix.

Each job installs its wheel into a new virtual environment.
It checks that VAD and decision inference load the bundled native libraries.
It runs the Python tests against the installed package with both native engines required.
Tests generate small synthetic checkpoints. No model is downloaded.
The release verifier checks PE and Mach-O architecture headers against wheel tags.
It preserves exact bundled model checksums.

For a targeted check, run `publish.yml` manually with `desktop_only` enabled.
Leave `publish` disabled. Desktop-only runs cannot publish a package.
The verified wheels are saved as `python-distributions-windows-x86_64`,
`python-distributions-macos-x86_64`, and `python-distributions-macos-arm64` artifacts.
Private engine source is not included in these artifacts.

## Publish from GitHub Actions

The workflow is `.github/workflows/publish.yml`. It builds and verifies the
portable wheel, source archive, Linux x86_64 and ARM64 native wheels,
Windows x64 wheel, and both macOS architecture wheels.
The current decision runtime supports MiniLM/BERT exports.
Its ARM64 support is experimental.
ARM64 builds in manylinux_2_28 on a native ARM64 runner.
x86_64 retains the glibc 2.34 floor. All native build jobs must pass before publishing. It tests the
native loader, Python API, and clean installations before saving the artifacts.
Models come exclusively from `src/akiki/models/`; no model is downloaded.

The training job also must pass before publishing.
It checks out `Akiki-AI/akiki-training` at `AKIKI_TRAINING_REV`.
It builds a private wheel under `verification/private-training/`.
It installs that wheel with the verified public x86_64 wheel and CPU runtimes.
It tests the training API, synthetic training, and native exports offline.
It checks the public archives again after the integration tests.
It does not upload the private wheel or companion source as a workflow artifact.
The publish job retrieves only the public distributions from the build jobs.
No manually built training wheel is required.

Run the workflow manually on a trusted change branch to build and test before merging.
Leave the `publish` input disabled.
Pushes to `main` also run these checks after reviewed changes are merged.
Only trusted builds can check out the private engine and training source.
Publishing a GitHub release runs the same checks and then uploads
the verified artifacts to PyPI using Trusted Publishing. The release tag must
be `v` followed by the version in `pyproject.toml`, for example `v0.2.2`. Update
all the version locations listed above first, and use a version that has not
already been uploaded to PyPI.
For a release candidate, mark the GitHub release as a prerelease.
Its package version must include the same `rc` suffix as its tag.

If publication fails before uploading, submit the workflow repair through a PR to `main`.
After merging, run the workflow manually on `main` with the `publish` input enabled.
The retry requires an existing GitHub release.
It checks that package source matches its version tag;
only the workflow and this release document may differ. This preserves the tag
while allowing CI-only repairs. All native build jobs run again before upload.

The existing PyPI `akiki` project must trust owner `Akiki-AI`, repository
`akiki-python`, workflow filename `publish.yml`, with the environment left
blank. Only the publishing job has `id-token: write`; no PyPI token is stored.

The native source is pinned by `AKIKI_INFERENCE_REV` in the workflow. Update
that commit deliberately when changing the engine. It must be reachable on
GitHub and have passed the required engine review. Its MiniLM C ABI
(`crates/akiki-minilm-engine/src/ffi.rs`) must match `src/akiki/decision.py`.
`akiki-python` is public. The engine source remains private.
Source checkout uses a dedicated read-only SSH deploy key:

1. Enable deploy keys in the organization's settings if they are disabled.
2. Add the public key under `akiki-inference` Settings > Deploy keys, leaving
   Allow write access unchecked.
3. Store the matching private key as the `AKIKI_INFERENCE_SSH_KEY` Actions
   secret in `akiki-python`.

The private key must never be committed. It is used only for source checkout
and is separate from PyPI Trusted Publishing. To rotate it, install a new
public key and update the secret before deleting the old deploy key.

Private training checkout uses a separate read-only SSH deploy key:

1. Add its public key under `akiki-training` Settings > Deploy keys.
   Leave Allow write access unchecked.
2. Store its private key as the `AKIKI_TRAINING_SSH_KEY` Actions secret in
   `akiki-python`.
3. Pin a tested training commit in `AKIKI_TRAINING_REV`.

Keep the training repository private. Do not add its source or wheel to `dist/`.
Do not add a dependency on the unpublished companion to public package metadata.

The x86_64 workflow targets glibc 2.34 and ARM64 targets glibc 2.28. Auditwheel must verify those floors;
a build that needs newer libc symbols fails rather than publishing a renamed
wheel. The engine's self-contained safetensors and configuration tests run in
CI; legacy engine tests that require external `.akv` fixtures are not part of
this workflow. The Python tests use the exact bundled release models.

## Publish from a local shell

Set `UV_PUBLISH_TOKEN` in your own shell or use configured trusted publishing.
Do not store credentials in this repository.

```bash
./publish.sh --dry-run
./publish.sh
```

The script verifies and uploads all wheels for the current version plus its
source archive. It requires the Linux, Windows, and macOS native wheels.
It rejects stale descriptions.
It does not upload artifacts from previous versions.
