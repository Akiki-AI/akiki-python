# Repository instructions

Follow [the Python style guide](docs/style.md) for code changes.

- Use plain enums for finite domain choices such as engines, precisions, modes,
  and statuses. Keep strings for free-form text and serialized representations.
- Use distinct enum types when parameters accept different sets of values.
- Run `ruff check .` and `ruff format --check .` before finishing Python changes.
  Run tests appropriate to the behavior changed.
- Preserve lazy imports for optional inference runtimes.

## Local model assets

- Never download models or model artifacts, including weights, tokenizers, and
  model configs. Use existing local caches. Check `/mnt/models/hub` and the
  project's configured model paths first.
- If a requested checkpoint is missing or ambiguous, ask for its local path.
  Do not fetch a checkpoint or silently substitute another model.
- Load Hugging Face models offline with `local_files_only=True` and
  `HF_HUB_OFFLINE=1`.
