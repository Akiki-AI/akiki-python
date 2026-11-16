# Repository instructions

Follow [the Python style guide](docs/style.md) for code changes.

- Use plain enums for finite domain choices such as engines, precisions, modes,
  and statuses. Keep strings for free-form text and serialized representations.
- Use distinct enum types when parameters accept different sets of values.
- Run `ruff check .` and `ruff format --check .` before finishing Python changes.
  Run tests appropriate to the behavior changed.
- Preserve lazy imports for optional inference runtimes.

## Documentation language

- Use ASD-STE100 Simplified Technical English for all documentation.
- Apply this rule to docstrings, notebooks, guides, and release instructions.
- Use short sentences, active voice, and one instruction per sentence.
- Use consistent technical terms. Preserve exact API names and code identifiers.
- Apply the same standard when you revise existing documentation.

## Ariadne terminology

- Describe Ariadne as adapting the query's representation so a frozen model can
  use its existing latent capabilities. The Ariadne layer does not contain
  learned skills; do not describe it as teaching the model new skills.
- Distinguish the Ariadne input affine from the separate classification head
  that reads out label probabilities.
- Ariadne applies to general-purpose LLMs and typically requires training fewer
  than 1% of the base model's parameter count. Distinguish the general technique
  from the checkpoint families supported by a particular implementation.

## Local hosting

- Keep websites, tutorials, demos, and generated artifacts local or on the
  user's own infrastructure. Do not host or deploy them on third-party
  services, including ChatGPT Sites.
- Deliver local HTML pages or applications. Keep their required assets local;
  do not add external hosting, publication, or deployment configuration.

## Local model assets

- Never download models or model artifacts, including weights, tokenizers, and
  model configs. Use existing local caches. Check `/mnt/models/hub` and the
  project's configured model paths first.
- If a requested checkpoint is missing or ambiguous, ask for its local path.
  Do not fetch a checkpoint or silently substitute another model.
- Load Hugging Face models offline with `local_files_only=True` and
  `HF_HUB_OFFLINE=1`.
