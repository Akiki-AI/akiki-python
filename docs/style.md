# Python style guide

Use explicit domain types and consistent formatting throughout the package,
tests, and developer tools. Ruff owns Python formatting and import ordering;
the settings in `pyproject.toml` are the source of truth.

## Documentation language

Use ASD-STE100 Simplified Technical English for all documentation.
This includes docstrings, notebooks, user guides, and release instructions.
Use short sentences and active voice. Give one instruction per sentence.
Use consistent technical terms. Keep exact API names and code identifiers.
Apply this standard when you revise existing text.

Name the primary inference engine the Akiki engine.
Use `Engine.AKIKI` in Python examples and `--engine akiki` in CLI examples.
ONNX and PyTorch are VAD compatibility fallbacks.
`Engine.AUTO` tries Akiki first, then ONNX, then PyTorch.
Do not use the implementation language as the engine name.

Describe the two primary interfaces as VAD and decision models.
Use `akiki.vad.VAD` and `akiki.decision.DecisionModel` in API overviews.
Describe MiniLM/BERT as the family supported by the current decision-model runtime.
Use `MiniLM` for direct access to that family.
Keep model discovery, training support, and inference support separate.
Successful checkpoint inspection does not establish inference support.

## Enums and domain types

Use a plain `enum.Enum` whenever a value represents a finite set of meaningful
choices, such as an inference engine, precision, mode, or status. Use enum
members for parameters, defaults, attributes, dispatch, and diagnostic keys.
Keep `str` for free-form text and external representations.

Use separate enum classes when parameters accept different sets of choices.
For example, VAD model precision and FFT precision have different domains:

```python
from akiki.vad import VAD, FFTPrecision, ModelPrecision

with VAD(
    model_precision=ModelPrecision.INT8,
    fft_precision=FFTPrecision.FP32,
) as detector:
    frames = detector.push([0.0] * 256)
```

VAD and decision models also have distinct model precision domains. Use
`akiki.vad.ModelPrecision` with `VAD` and `akiki.decision.ModelPrecision` with
`DecisionModel` or `MiniLM`.
Do not share enum types just because some serialized values match.

Validate enum types at public API boundaries. Reject strings and members of a
different enum with `TypeError`; reject unsupported combinations of otherwise
valid values with `ValueError`. Keep engine compatibility checks separate from
the precision types. Compare enum members with `is`.

Convert between enum members and strings at CLI, JSON, metadata, and native
library boundaries. Derive CLI choices from the enum rather than maintaining a
second list of strings. Keep native format IDs explicit so reordering an enum
cannot change the binary format. Use `None` for an omitted optional setting
when it means to resolve the engine's default.

Whether a value needs an enum is a design decision checked during review;
Ruff does not infer domain types from strings.

## Formatting and imports

- Use four spaces for indentation, double quotes, and LF line endings.
- Use a 100-character target line length. Let Ruff wrap expressions and calls;
  wrap prose manually where useful.
- Keep two blank lines between module-level definitions and one between class
  methods. Let the formatter manage expression spacing and trailing commas.
- Group imports as standard library, third party, then Akiki, with a blank line
  between groups. Let Ruff sort within each group.
- Keep ordinary imports at module scope. Import optional runtimes inside the
  functions that need them so help, version output, and other engines remain
  usable without those dependencies.
- Target Python 3.10 or later. Use built-in collection annotations and `X | None`.
  Add `from __future__ import annotations` only when deferred evaluation is
  needed, such as for a forward reference or a type-only import.
- Use named functions for assigned callables; reserve lambdas for short inline
  callbacks.

## Development tools

Install the development dependency group into an activated virtual environment.
The pip command requires pip 25.1 or later:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[onnx]' --group dev
pre-commit install
```

With uv, use `uv pip install -e '.[onnx]' --group dev` instead of the pip install
commands. The ONNX extra lets the VAD tests run without a local Akiki library.
Model files stay local; do not download checkpoints for tests.

Apply safe lint fixes and formatting before reviewing a change:

```bash
ruff check --fix .
ruff format .
```

Run the checks used in CI:

```bash
ruff check .
ruff format --check .
yamllint --strict .github .pre-commit-config.yaml .yamllint.yaml
HF_HUB_OFFLINE=1 python -m pytest -q -rs
```

Check the executable examples in the public inference docstrings:

```bash
HF_HUB_OFFLINE=1 python -m pytest -q --doctest-modules \
    src/akiki/__init__.py src/akiki/vad.py src/akiki/decision.py src/akiki/hub.py
```

VAD examples require a working local engine. Examples with placeholder text-model
paths are skipped. The normal suite uses synthetic local checkpoints for native
text tests. Real-model tests require an explicit `AKIKI_MINILM_MODEL` path.

Training tests require the private companion and its Ariadne dependencies.
Install it from an authorized local checkout or wheel.
The `training` extra supplies runtimes only. The base development group installs neither component.
Use the test summary to identify skipped optional-runtime checks.
Passing inference tests does not establish that the optional training tests ran.

The commit hooks run Ruff fixes, Ruff formatting, then YAML linting. Use
`pre-commit run --all-files` to check tracked files before committing. Generated
build outputs, model caches, and verification environments are excluded from
Python style checks. Ruff is pinned in the development group and commit hooks;
update those pins and `tool.ruff.required-version` together.

See the official [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
and [formatter documentation](https://docs.astral.sh/ruff/formatter/) for tool
behavior, and the [pip dependency group documentation](https://pip.pypa.io/en/stable/user_guide/#dependency-groups)
for installation options.
