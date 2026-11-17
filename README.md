# Akiki

Akiki provides local voice activity detection (VAD) and decision models.
VAD returns a speech probability, a speech/silence decision, and timestamps for
each 16 ms frame. Decision models classify text with saved specialists and
return label probabilities through [`DecisionModel`](#decision-models).
Training uses an optional companion package. An experimental teacher API can
request example proposals from a server on your own computer or infrastructure.

Decision models use a frozen base checkpoint and specialist adapters.
The Akiki engine currently supports MiniLM/BERT exports for decision inference.
Support for other model families requires additional inference implementations.

The VAD model is included in the package. VAD runs locally, without a model
download, network connection, or external service. The Akiki engine uses INT8
model inference by default, with an FP32 FFT.

## Install

Requires Python 3.10 or later.

This README describes the current source tree. For an installed release, use
the README from its matching version tag.

```bash
pip install akiki
```

Or add it to a uv project:

```bash
uv add akiki
```

The base installation uses the Akiki engine on supported hosts.
ONNX and PyTorch provide compatibility fallbacks for VAD.
Install `akiki[onnx]` or `akiki[pytorch]` when you need a compatibility engine.

### Windows and macOS

Native wheels support Windows x64, macOS Intel, and macOS Apple Silicon.
Each wheel includes the Akiki engine for VAD and decision models.
Pip selects the wheel for your Python installation. No engine build is required.
Windows ARM64 wheels are not available.
The macOS wheels target macOS 11.0 or later.

#### Windows (PowerShell)

Use 64-bit x86 Python 3.10 or later.
Run these commands in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install akiki
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m akiki vad ".\recording.wav" --engine akiki
.\.venv\Scripts\python.exe -m akiki vad --help
```

Use a mono, 16,000 Hz WAV file for `recording.wav`.
To use an existing local MiniLM/BERT classifier, run:

```powershell
.\.venv\Scripts\python.exe -m akiki minilm --model "C:\models\my-classifier" "Please save my changes."
```

Replace `C:\models\my-classifier` with your local export directory.
The `minilm` command gives direct access to the current decision-model runtime.
Use [`DecisionModel`](#decision-models) to manage saved specialists through Python.

#### macOS (Terminal)

Use Python 3.10 or later.
Use a Python installation that matches your Mac's architecture.
Run these commands in Terminal:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install akiki
export HF_HUB_OFFLINE=1
.venv/bin/python -m akiki vad "./recording.wav" --engine akiki
.venv/bin/python -m akiki vad --help
```

Use a mono, 16,000 Hz WAV file for `recording.wav`.
To use an existing local MiniLM/BERT classifier, run:

```bash
.venv/bin/python -m akiki minilm --model "./models/my-classifier" "Please save my changes."
```

Replace `./models/my-classifier` with your local export directory.
The Python VAD and decision-model APIs below are the same on all supported systems.

#### VAD compatibility fallback

If a native wheel is unavailable, use the ONNX compatibility fallback for VAD.
Create a virtual environment as shown in your platform example above.
On Windows, run:

```powershell
.\.venv\Scripts\python.exe -m pip install "akiki[onnx]"
.\.venv\Scripts\python.exe -m akiki vad ".\recording.wav" --engine onnx
```

On macOS, run:

```bash
.venv/bin/python -m pip install 'akiki[onnx]'
.venv/bin/python -m akiki vad "./recording.wav" --engine onnx
```

This fallback uses the bundled VAD model. It does not enable decision-model inference.
PyTorch is the second VAD compatibility fallback.

## Python usage

Use `akiki.vad` for VAD and `akiki.decision` for decision models.
The training and teacher modules support the decision-model workflow:

| Module | Main interface | Supporting types |
| --- | --- | --- |
| `akiki.vad` | `VAD` for speech detection | `SpeechFrame`, `Engine`, `ModelPrecision`, `FFTPrecision` |
| `akiki.decision` | `DecisionModel` for specialist decisions | `BaseModel`, `Specialist`, `AffineLayer`, `TrainedSpecialist`, `SpecialistEvaluation`, `TrainingAdvice`, `list_base_models`, `Classification`, `ModelPrecision` |
| `akiki.training` | Classifier training with a private companion | The installed companion's API, including `LocalModel`, `ModelFamily`, and `Trainer` |
| `akiki.teacher` | Experimental review through a local or self-hosted server | `Teacher`, `Verdict`, `DecisionReview`, `TeachingResult` |

Import the primary interfaces from their modules:

```python
from akiki.vad import VAD
from akiki.decision import DecisionModel, Specialist
```

Import configuration enums and result types from the relevant module.
VAD and decision inference have distinct `ModelPrecision` enums.
Their supported formats differ.
`MiniLM` is the lower-level interface for direct embeddings and classification
with the currently supported MiniLM/BERT family.
The existing `akiki.VAD` and `akiki.MiniLM` shortcuts remain available.
The existing top-level result types and VAD enums remain available as
compatibility aliases. Import `training` explicitly when its optional companion
package is installed; the base package does not load it.
`dir(akiki)` shows the main classes, modules, and version. The `training` and
`teacher` modules appear after explicit import. Supporting aliases and internal
modules remain accessible but are omitted from that list.

The decision-model module is named `akiki.decision`. Existing imports from
`akiki.minilm` remain available for compatibility and use the same classes
and precision enum. Named top-level compatibility imports remain available.
Wildcard imports expose only `VAD`, `MiniLM`, `vad`, `decision`, and `__version__`.

## Voice activity detection (VAD)

Akiki accepts mono audio sampled at 16,000 Hz.

```python
import soundfile as sf
from akiki import VAD

audio, sample_rate = sf.read("recording.wav", dtype="float32")

with VAD() as vad:
    for frame in vad.push(audio, sample_rate=sample_rate):
        print(frame.start_seconds, frame.end_seconds, frame.probability, frame.speech)
```

For live audio, call `vad.push()` repeatedly with successive audio buffers.
Buffers can have any length; incomplete frames are retained for the next call.
Call `vad.reset()` before starting a new recording. A final incomplete frame is
not padded or scored.

Use `VAD(threshold=0.6)` to set the speech decision threshold. Every frame still
includes its probability, regardless of the threshold.

The public API includes docstrings for editor hints and interactive help:

```python
import akiki

help(akiki)  # Overview of vad, decision, training, and teacher
help(akiki.vad)  # Speech detection, configuration, and result types
help(akiki.decision)  # Decision models, specialists, and result types
help(akiki.decision.DecisionModel)  # Base-model selection and specialist inference
help(akiki.decision.Specialist.train)  # Training parameters and saved exports
help(akiki.VAD)  # Configuration, attributes, and streaming example
help(akiki.VAD.__init__)  # Constructor parameters and defaults
help(akiki.VAD.push)  # Audio input, buffering, results, and errors
help(akiki.vad.SpeechFrame)  # Result fields and timestamp units
```

### Command line

```bash
akiki vad recording.wav
akiki vad recording.wav --threshold 0.6
akiki vad --help
```

The command prints one JSON object per complete frame, containing its timestamps,
speech probability, and speech decision.

VAD options follow the `vad` subcommand. The `akiki` command has three
subcommands: `vad`, `minilm`, and `download`. Use `akiki --help` to list them.
The equivalent module invocation is `python -m akiki vad recording.wav`.
Specialist management and teacher review use the Python API. The optional
training command belongs to the separate companion package.
The `minilm` command gives direct access to the current MiniLM/BERT runtime.
It does not manage decision-model specialists.

### Inference engines

The Akiki engine is the primary engine for VAD and decision-model inference.
Its current decision-model implementation supports MiniLM/BERT exports.
ONNX and PyTorch are VAD compatibility fallbacks.

Linux x86_64 wheels include the Akiki engine for systems with glibc 2.34 or later.
Pip and uv select a compatible wheel automatically. No engine build is required.
For Windows and macOS, see the [platform installation examples](#windows-and-macos).

Version **0.1.2** includes **Linux ARM64 VAD support**, including Raspberry Pi 5
with a 64-bit OS and glibc 2.28 or later. This wheel includes the Akiki VAD library
and supports all six VAD precision modes with the same INT8 default. INT8
projections automatically use KleidiAI Neon DotProd kernels on compatible CPUs,
with a portable fallback on older ARM64 CPUs. Other precisions retain portable
execution. The optimized path has been tested on a physical Raspberry Pi 5.

Version **0.1.4** also includes the Akiki MiniLM engine in the Linux ARM64 wheel.
**ARM64 MiniLM support is experimental**, including inference with local Ariadne
adapters. It uses the existing MiniLM API and requires glibc 2.28 or later.

`VAD(engine=Engine.AUTO)` tries the Akiki engine first.
If it is unavailable, selection tries ONNX, then PyTorch.
The compatibility engines are optional. The base package requires neither runtime.

To install the ONNX compatibility engine:

```bash
pip install 'akiki[onnx]'
# or: uv add 'akiki[onnx]'
```

To install the PyTorch compatibility engine:

```bash
pip install 'akiki[pytorch]'
# or: uv add 'akiki[pytorch]'
```

The PyTorch extra does not install ONNX Runtime. Install `akiki[onnx,pytorch]`
if you want both alternatives. If no engine is available, Akiki raises an error
with installation instructions.

Import `Engine` from `akiki` and choose an engine with `VAD(engine=Engine.AKIKI)`,
`VAD(engine=Engine.ONNX)`, or `VAD(engine=Engine.PYTORCH)`. The default is
`Engine.AUTO`; the Python API requires enum members. On the command line, use
`akiki vad recording.wav --engine akiki`, `--engine onnx`, or `--engine pytorch`.
An explicitly selected engine raises an error if it is unavailable. Check
`vad.engine` to see the active `Engine` member. `vad.engine_errors` maps
`Engine` members to the reasons they were unavailable during automatic selection.

All VAD engines use the same bundled model and streaming API.
The compatibility fallbacks apply to VAD.
Decision-model inference currently requires the Akiki engine and a supported export.

### Model precision

`VAD()` and the command line use INT8 by default when running the Akiki engine.
The package generates the INT8 weights from the bundled FP32 model on first use
and reuses them from the local cache. The FFT remains FP32.

Choose another precision when creating the detector. For example, to use the
original FP32 weights:

```python
from akiki.vad import VAD, Engine, ModelPrecision

with VAD(engine=Engine.AKIKI, model_precision=ModelPrecision.FP32) as vad:
    frames = vad.push(audio, sample_rate=sample_rate)
```

The command-line equivalent is:

```bash
akiki vad recording.wav --model-precision fp32
```

| `model_precision` | Convolution and GRU projections |
| --- | --- |
| `ModelPrecision.INT8` (Akiki default) | INT8 weights and dynamically quantized activations |
| `ModelPrecision.FP32` | Original FP32 weights and activations |
| `ModelPrecision.FP16` | FP16 weights, FP32 activations and accumulation |
| `ModelPrecision.BF16` | BF16 weights and projection activations, FP32 accumulation |
| `ModelPrecision.INT4` | Packed INT4 weights and dynamically quantized INT8 activations |
| `ModelPrecision.INT8_FP32_INPUT` | INT8 weights; first projection keeps FP32 activations |

The Python VAD API requires `ModelPrecision` and `FFTPrecision` enum members,
both exported from `akiki.vad`. Omit `model_precision` or pass `None` to use the
selected engine's default. CLI precision options retain their lowercase names.

The FFT stays FP32: `fft_precision` accepts only `FFTPrecision.FP32`. Recurrent state,
nonlinearities, biases, and the output head also remain FP32. Lower precision
can change speech probabilities; speed depends on the CPU and chosen format.

Conversion runs locally using the included weights, without downloading models
or installing PyTorch or ONNX Runtime. Generated variants are reused from the
user cache. Set `cache_dir` or `AKIKI_CACHE_DIR` to choose another location. The
original weights are preserved; incomplete or corrupt cache entries are rebuilt.

Check `vad.model_precision` and `vad.fft_precision` to inspect the enum settings.
When no model precision is specified, ONNX and PyTorch use FP32, including when
`engine=Engine.AUTO` falls back to either runtime. Explicit precision requests are
always honored: non-FP32 precision requires the Akiki engine and raises an error if a
compatible Akiki engine is unavailable.

### Precision and performance

INT8 is the default because it gave the best measured speed and average
precision tradeoff in our benchmark. It took **8.90 µs per frame through
`VAD.push()`**, compared with **16.03 µs for FP32**, with comparable average
precision on both TEN and VoxConverse. INT4 used less weight storage but was
slower and slightly less accurate than INT8 on this CPU.

| Model precision | Native call (µs) | Python `VAD.push()` (µs) | TEN AP (%) | VoxConverse AP (%) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 9.68 | 16.03 | 98.019 | 99.304 |
| FP16 | 7.40 | 12.80 | 98.019 | 99.304 |
| BF16 | 5.45 | 10.73 | 98.017 | 99.304 |
| **INT8 (default)** | **3.78** | **8.90** | **98.042** | **99.311** |
| INT4 | 6.61 | 11.74 | 97.841 | 99.287 |
| INT8 with FP32 input | 5.52 | 10.73 | 98.009 | 99.306 |

These measurements use Akiki 0.0.5 on an AMD Ryzen 9 9900X, one pinned CPU core,
and an FP32 FFT for every precision. Latency is the median of three fresh runs,
each with 2,000 warmup calls and 7,500 timed calls, processing one 16 ms audio
frame at a time. Native timing includes a call through `ctypes` with prepared
pointers. Python timing also includes input validation, buffering, state
handling, timestamps, and frame output. Model loading, first-use conversion,
audio decoding, and the time needed to receive audio are excluded. Results
depend on the host CPU and workload.

Average precision (AP) summarizes the precision–recall curve across thresholds;
higher is better. Scores use 30 TEN recordings and 48 VoxConverse recordings,
averaging pairs of 16 ms predictions against the existing 32 ms labels, without
time shifts or smoothing. These are development benchmarks: the small AP
differences do not establish a statistically significant accuracy improvement.

## Decision models

`DecisionModel` is the primary interface for text decisions.
A decision model combines a frozen base checkpoint with saved specialists.
Each specialist has its own intents, examples, input affine, and classification head.
The decision-model interface describes these objects independently of a model family.
Inference support depends on the engine implementation for the selected family.
The current Akiki engine supports MiniLM/BERT exports.

### Base-model selection

`DecisionModel` selects a local text checkpoint through its `base_model`
property and classifies saved specialists with the native Akiki engine.
`list_base_models()` returns `BaseModel` descriptors for
encoder snapshots already in local Hugging Face caches. Each exposes `name`,
`path`, `vocab_size`, `embedding_dim`, and `embedding_shape`:

```python
from akiki.decision import DecisionModel, list_base_models

for base in list_base_models():
    print(base.name, base.path, base.embedding_shape)

# Select the exact local checkpoint that you intend to use.
model = DecisionModel("path/to/local/base-model")
print(model.base_model.embedding_shape)  # (vocabulary rows, token vector width)
width = model.base_model.embedding_dim
print((width, width), (width,))  # Input affine weight and bias shapes.
```

You can also pass `base_model` to the constructor, assign a local checkpoint
path, or use an exact cached repository name when only one revision is present.
Assign `None` to clear the selection. Multiple revisions require an explicit
path. The selected `base_model` is a descriptor; use `.path` for its checkpoint
directory. Descriptors can also be passed to filesystem APIs directly.
Selection validates metadata and required files; it does not load a model.
Successful selection does not establish compatibility with a training or inference engine.

| Operation | Checkpoint support |
| --- | --- |
| `list_base_models()` | Recognized encoder snapshots with complete local files |
| `DecisionModel.base_model` | Local text checkpoints whose metadata and input dimensions can be inspected |
| `Specialist.train()` | Training families supported by the installed companion package |
| `DecisionModel` inference | Currently supported MiniLM/BERT specialist exports with `model.safetensors` and `tokenizer.json` |
| `MiniLM` | Direct embeddings and classification for the same supported family |

Discovery can find other encoder families and sharded checkpoints.
The native text loader requires one unsharded `model.safetensors` file.
Qwen specialist inference is not exposed by this Python binding.

### Specialists

Each `DecisionModel` also has a `Specialists` property containing its own empty
`set[Specialist]`. Each specialist has an `intents: set[str]` and an
`examples: dict[str, str]` mapping example text to its intent. Both default to
empty collections belonging to that instance:

```python
from akiki.decision import Specialist

specialist = Specialist(
    intents={"question", "task"},
    examples={
        "What time is it?": "question",
        "Restart the server.": "task",
    },
)
model.Specialists.add(specialist)
```

Each specialist has a persistent `affine_layers` mapping. An `AffineLayer`
contains the trained input affine's NumPy `weight` and `bias` arrays. Entries are
keyed by the exact base-model descriptor, so different checkpoint paths or
revisions remain separate even if their embedding dimensions match:

```python
from akiki.decision import AffineLayer

# After training produces the input affine parameters, assignment saves them:
specialist.affine_layers[model.base_model] = AffineLayer(weight, bias)
affine = specialist.affine_layers.get(model.base_model)
```

The cache writes local NumPy archives atomically and loads them on lookup.
Recreating a specialist with the same intents and examples reuses its saved
affines, including across Python sessions. The input affine remains separate
from the classification head. Cache keys include the base-model path and
embedding dimensions, plus a fingerprint of the specialist's intents and
examples. Editing those collections selects another namespace; previous trained
layers stay on disk and become available again if the original data is restored.
Cache identity uses the base path and embedding dimensions, not a weight checksum.
Keep checkpoint files unchanged. Use a separate path for each checkpoint revision.

Pass `Specialist(..., cache_dir="path/to/affine-cache")` to choose the location.
The default is `affines/` under `AKIKI_CACHE_DIR`, or the platform's Akiki user
cache directory. Construction with no initial layers does not write files or load layers.
Supplying `affine_layers=` saves those entries during construction.
Loaded arrays can be edited in memory; assign the layer back to the mapping to
save changes. Missing entries return `None` through `.get()`. Damaged entries
raise an error and are preserved instead of being treated as a reason to retrain.
The affine mapping contains no classification head. Assigning an affine alone
does not create the complete export required by `DecisionModel`.

Input embedding table dimensions come from safetensors headers when available,
with a configuration fallback for other weight formats. They are read from the
selected checkpoint at runtime; there is no catalog of model dimensions.
Dimension inspection and explicit selection are independent of the encoder
families currently included in discovery. Factorized encoders
can have a token embedding width smaller than their hidden width. Sentence
embedding output dimensions can also differ. An input affine preserving a
token width of `d` needs a `d × d` weight matrix and `d` biases; the
classification head is separate.

Discovery checks the configured Hugging Face caches and the system model cache,
if present.
Pass `list_base_models(cache_dir="path/to/hub")` to search a specific cache.
It skips incomplete snapshots and generative models. Discovery and selection
use local files only and require no optional training or Hugging Face packages.

### Specialist training and saved exports

`Specialist.train()` calls the separately installed private `akiki-training` companion
and reuses a complete saved adapter when its base, examples, holdouts, and
training configuration match. The saved artifact retains both the input affine
and the separate classification head, with the label order and training report:

```python
from akiki.decision import Specialist
from akiki.training import TrainingConfig

specialist = Specialist(
    intents={"question", "task"},
    examples={
        "What time is it?": "question",
        "Where is the station?": "question",
        "Restart the server.": "task",
        "Save this document.": "task",
    },
    cache_dir=".cache/specialists",
)
config = TrainingConfig(epochs=20, minimum_epochs=4)
trained = specialist.train(model.base_model, config=config)
report = specialist.eval()
report  # Displays a per-intent table in a notebook.
```

Labels in `TrainedSpecialist.labels` are sorted by intent name.
Training requires the private companion even when it reuses a saved result.
Saved-report evaluation and native inference do not require the companion.

### Decision inference

Classify with the saved specialist through `DecisionModel`.
The current implementation loads compatible MiniLM/BERT exports through the Akiki engine.
The result includes the predicted `label`, its
`index`, and `probabilities` and `logits` in `trained.labels` order:

```python
from akiki.decision import DecisionModel, ModelPrecision

with DecisionModel(
    model.base_model,
    model_precision=ModelPrecision.FP32,
    threads=4,
) as decisions:
    decisions.Specialists.add(specialist)
    result = decisions.classify("Please save my changes.")
    print(result.label, dict(zip(trained.labels, result.probabilities)))

    # Select explicitly when more than one specialist is attached:
    result = decisions.classify("Where is the station?", specialist=specialist)

    # Or get a separate classification from every attached specialist:
    results = decisions.decide("Please save my changes.")
    print(results[specialist].label)
```

Inference supports native MiniLM/BERT specialist exports. Qwen specialist
inference is not yet exposed by the Python binding. It needs no training
package, PyTorch, Transformers, or network access; an equivalent `Specialist`
recreated with the same intents, examples, and cache directory finds the saved
export after a restart. Missing, changed-data, or damaged exports raise errors
without starting training. `classify()` uses the only attached specialist when
one exists; otherwise select one explicitly. `decide()` runs all attached
specialists sequentially and keeps their results separate.

Construction loads no native models. First inference loads a handle per
specialist; subsequent calls reuse it. A new saved training run replaces the
old handle on its next use. Changing the selected base or replacing the
`Specialists` set releases unused handles; mutations such as `.remove()` are
noticed on the next inference call. Use a context manager or `close()` to
release all native resources. Inference calls on one `DecisionModel` are serialized.
Do not change specialist data or its attached set during inference.

The current Akiki engine folds each input affine into its own embedding table
in memory, so different specialists currently load separate base copies. Saved
checkpoint and adapter files stay unchanged. `model_precision` defaults to
`ModelPrecision.INT8`. Use `FP32` for comparisons with training metrics.
`library_path=` selects an existing native shared library explicitly.
Linux ARM64 MiniLM support remains experimental. Inputs are passed verbatim and
are never silently truncated.

### Reports and training advice

`Specialist.eval()` reads the saved validation report. It does not run new
predictions or evaluate a new dataset. Test scores are not part of this report.
With several saved base models, pass `base_model=` unless the instance already
has a selected training result.

`report.per_intent` maps each intent to validation accuracy, cross entropy,
training support and loss, the selected adapter's input-affine and head
gradient norms, a recent validation-loss slope, and a `TrainingAdvice` enum.
Gradients are norms of mean per-intent training cross entropy before clipping;
loss slopes are changes in held-out cross entropy per epoch. Suggestions are
heuristics, and small holdouts are identified as insufficient evidence rather
than a reliable plateau. Supply `validation_examples=` and optionally
`test_examples=` to use explicit disjoint holdouts; test examples are never
used for optimization or checkpoint selection.

Use `continue_training=True` to start from saved affine/head weights with a fresh
optimizer. Automatic holdouts retain the original examples; explicit holdouts
must be supplied unchanged. After changing examples across sessions, use
`resume_from=trained` or the earlier export directory. Existing successful
exports are preserved, and continuation keeps its starting checkpoint if no
epoch improves validation loss. Complete specialist exports reference the
local frozen base by symlink instead of copying its weights for every run.
Keep that base at its original path. These specialist exports depend on it.

An optional local teacher can propose additional samples when an intent's
validation performance has flattened:

```python
from akiki.training import LocalTeacher

weak_intents = report.intents_needing_examples
if weak_intents:
    proposed = specialist.suggest_examples(
        LocalTeacher("path/to/existing/local/teacher"),
        intents=weak_intents,
        examples_per_intent=10,
    )
    # Review proposed text/intent pairs before accepting them.
    specialist.examples.update(proposed)
    trained = specialist.train(model.base_model, config=config, continue_training=True)
```

`LocalTeacher` requires an explicit existing local checkpoint and uses offline loading.
Generation never changes the specialist's examples automatically. Proposed
samples are deduplicated against known examples and holdouts. Use
`excluded_examples=` to exclude any additional reserved data. A custom local
teacher can implement `generate_examples(...)` with the same dictionary output.
Evaluation of a saved result works without loading a training runtime.
The local-teacher API sends `specialist.examples` as reference data.
Use separate explicit validation and test dictionaries to keep holdouts out
of those references. Custom teachers control their own model and network access.
`Specialist.suggest_examples()` requires the private training companion,
including when a custom teacher is supplied.

### Teacher review

An experimental server-backed teacher can review the decisions of an attached,
trained specialist and propose similar examples when it judges a decision to be
incorrect:

```python
from akiki.teacher import Teacher, Verdict

teacher = Teacher(
    "http://localhost:8000/v1",
    model="your-served-teacher-model",
    # api_key="...",  # Only if your server requires authentication.
    task_description="Distinguish requests to perform an action from questions.",
    intent_descriptions={
        "question": "The user wants information or an explanation.",
        "task": "The user asks for an action, including polite question forms.",
    },
)

with DecisionModel(model.base_model, model_precision=ModelPrecision.FP32) as decisions:
    decisions.Specialists.add(specialist)
    feedback = teacher.review(
        ["Could you save this document?", "How does autosave work?"],
        decisions,
        specialist=specialist,
        examples_per_error=5,
        # Exclude any additional reserved datasets (mapping keys are accepted).
        excluded_examples={"Where can I find my saved documents?": "question"},
    )

for review in feedback.reviews:
    print(review.text, review.decision.label, review.verdict.value, review.reason)
    if review.verdict is Verdict.INCORRECT:
        print("Teacher correction:", review.expected_label, review.examples)

# Export proposals without overwriting an existing file.
feedback.save_examples("supplemental.jsonl")

# After reviewing and accepting the proposals:
if feedback.examples:
    specialist.examples.update(feedback.examples)
    trained = specialist.train(model.base_model, config=config, continue_training=True)
```

`Teacher` uses the server's Chat Completions endpoint through the Python standard
library; it needs no SDK or training extra. A server root selects `/v1`; a URL
with a path is used as the API base. Omit `model=` only when the server's `/models`
endpoint lists exactly one model. For servers without JSON response mode, set
`json_mode=False`; responses must still follow the requested JSON structure.
Supply a task policy and label descriptions where label names alone are unclear.

Each input receives a correct, incorrect, or uncertain `Verdict` with a reason.
Only incorrect decisions trigger a second request for examples with the corrected
label. Proposals exclude the review inputs, training data, known validation/test
texts, explicitly reserved data, and duplicates across the batch. Matching
normalizes case, Unicode compatibility characters, and whitespace. Reserved
inputs are rejected before requests, and reserved examples are omitted from
prompt references. Use new or training inputs for feedback; keep final test data
separate. These filters detect text duplicates, not semantic paraphrases.

Teacher requests send input texts, labels, the policy, and up to three training
examples per label to the supplied server. No model files are downloaded. Invalid
responses, contradictory judgements, and request failures raise `TeacherError`;
neither the specialist nor its training data is changed. The returned labels are
teacher proposals and need review before acceptance. JSONL export uses `text` and
`label` fields compatible with `akiki.training.load_examples`. When continuing a
run with explicit validation/test data, supply the original holdouts to `train()`.

### Current inference support: MiniLM/BERT

The current Akiki engine implements MiniLM/BERT inference for decision models.
`DecisionModel` uses this implementation to run saved specialists.
The lower-level `akiki.decision.MiniLM` class gives direct access to embeddings
and classification for this family. It loads plain checkpoints, such as
`sentence-transformers/all-MiniLM-L6-v2`, and Akiki Ariadne variants.
These variants add a small adapter to the frozen encoder. One example is
the Akiki MiniLM-L6-v2 Clover intent classifier.

To train a compatible classifier from labelled examples, use the separate
`akiki-training` project. Its compatible MiniLM/BERT exports load through the
same `MiniLM` API. Other training families need their own inference runtime.

The MiniLM weights are not included in the package. Pass the path to a local
copy when creating the model:

```python
from akiki.decision import MiniLM

with MiniLM("path/to/all-MiniLM-L6-v2") as model:
    vectors = model.embed(["A man is eating food.", "Someone is eating a meal."])
    print(vectors[0] @ vectors[1])  # Cosine similarity.
```

Embeddings are float32 NumPy arrays, mean-pooled and L2-normalized, so their dot
products are cosine similarities. `embed()` accepts one string or an iterable
of strings. Multiple texts are encoded one at a time.

Variants with a head classify text in their own input format:

```python
with MiniLM("path/to/Akiki-MiniLM-L6-v2-clover-intent") as model:
    result = model.classify("Current user message: Restart my web server.")
    print(result.label, result.probabilities[result.index])

    follow_up = model.classify(
        "Conversation context:\n"
        "assistant: Would you like me to check whether last night's backup ran?\n\n"
        "Current user message: yes"
    )
    print(follow_up.label)
```

Each classification has the most probable output's `index` and `label`, the
`probabilities` of every output, and the raw `logits`. Probabilities are a plain
softmax, without calibration.
`label` is `None` when the head has no label names in `model_config.json`.

A model folder is either a checkpoint with `config.json`, `model.safetensors`,
and `tokenizer.json`, or a variant folder holding that checkpoint in
`base_model/` beside `adapter.safetensors`. A variant's optional
`model_config.json` names its outputs (`labels`) and sets a chunk length for long
text (`max_length`). Pass `adapter_path` to use an adapter stored elsewhere, or a
variant's `base_model` folder to use the plain encoder.

The sequence limit is `model.max_tokens`, including special tokens.
It comes from the loaded checkpoint. Longer text raises `ValueError` unless
chunking is enabled. Inputs are never silently truncated.
`chunk_tokens` must be from 3 to `model.max_tokens`. It requires a tokenizer
with exactly two special tokens. Chunk embeddings are normalized, combined by
a mean weighted by their text-token counts, then normalized again.
The classification head is applied to the combined embedding.

A variant's integer `max_length` enables chunking automatically.
Passing `chunk_tokens=None` retains that setting.
When comparing embeddings with another toolkit, match its token limit and precision.

For task-specific inputs, such as typed decisions, `tokenize()` and
`encode_chunks()` give direct control over the chunks and their weights;
`encode()` returns both the embedding and any head logits.
Native calls on one `MiniLM` instance are protected by a lock.
An `embed()` call with several texts can be interleaved with other calls
between texts. Keep the instance open until all worker calls have finished.
`tokenize()` returns all IDs without padding or truncation, including sequences
longer than `model.max_tokens`. `encode_chunks()` requires complete token
sequences and positive finite weights. It adds no special tokens.

MiniLM requires the Akiki engine included in native wheels.
These wheels support Linux x86_64 and ARM64, Windows x64, and macOS Intel and Apple Silicon.
See the [Windows and macOS examples](#windows-and-macos) for current wheel availability.
**Linux ARM64 MiniLM support is experimental**. It has no ONNX or PyTorch alternative.
It uses up to four CPU threads by default; set `threads` to change this.
Projections use `akiki.decision.ModelPrecision.INT8` by default. To use the
original weights:

```python
from akiki import decision

with decision.MiniLM(
    "path/to/all-MiniLM-L6-v2", model_precision=decision.ModelPrecision.FP32
) as model:
    vector = model.embed("A sentence to encode.")
```

The other MiniLM formats are `FP16`, `BF16`, `INT8_FP32_INPUT`, `INT4`, and
`INT4_G32`. The Python API requires members of `akiki.decision.ModelPrecision`
and returns them through `model.model_precision`; precision strings and VAD
precision members are rejected with `TypeError`. Replace earlier
`model_precision="fp32"` calls with `model_precision=decision.ModelPrecision.FP32`.
CLI precision names stay lowercase, such as `--model-precision fp32`.

Weights are converted in memory when the model loads; the files are unchanged.
Attention, normalization, and pooling remain FP32. Lower precision uses less
memory but can change embeddings, probabilities and, for borderline inputs,
the label.

On the command line:

```bash
akiki minilm --model path/to/all-MiniLM-L6-v2 "A man is eating food."
akiki minilm --model path/to/Akiki-MiniLM-L6-v2-clover-intent \
    "Current user message: Restart my web server."
akiki minilm --help
```

The command prints JSON: the classification for models with a head, and the
embedding otherwise. Add `--embedding` to include the embedding as well.

#### Hugging Face models

Use the optional Hub integration with compatible MiniLM checkpoints or Ariadne
variants. The examples below use existing cache files:

```bash
pip install 'akiki[huggingface]'
```

```python
import os

from akiki.decision import MiniLM
from akiki.hub import download_model

os.environ["HF_HUB_OFFLINE"] = "1"

with MiniLM.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
) as model:
    vector = model.embed("A man is eating food.")

# Find a cached model without loading an inference engine.
path = download_model("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
with MiniLM(path) as model:
    vector = model.embed("Someone is eating a meal.")
```

Both functions accept `revision` (a branch, tag, or full commit hash),
`cache_dir`, `local_dir`, `local_files_only`, and `token`. `from_pretrained()`
also accepts the usual MiniLM constructor options, including `threads`,
`model_precision`, and `chunk_tokens`. Pin `revision` to a full commit hash
when you need the same model files across runs.

The downloader selects the supported model files, model card, and license
notices. It skips alternate weight formats, training data, and repository code.
The repository must contain the checkpoint or variant layout described above;
downloading another architecture does not make it compatible with MiniLM.
The same native engine and platform requirements apply after downloading.

On the command line:

```bash
export HF_HUB_OFFLINE=1

# Run an existing cached model.
akiki minilm --repo sentence-transformers/all-MiniLM-L6-v2 --local-files-only "Hello."

# Print the path of an existing cached model without loading it.
akiki download sentence-transformers/all-MiniLM-L6-v2 --local-files-only
```

Downloads use the [Hugging Face Hub cache and revision handling](https://huggingface.co/docs/huggingface_hub/guides/download).
Private or gated repositories use `HF_TOKEN` or your saved Hugging Face login;
the Python API also accepts `token`. Set `HF_HUB_CACHE` or pass `cache_dir` to
choose a cache location. `local_files_only=True` or `--local-files-only` requires
an existing local snapshot; missing required files produce an error.
`HF_HUB_OFFLINE=1` also forces local-only access and is never overridden.
Ordinary `MiniLM(path)` loading and inference do not contact Hugging Face.
Without the offline setting or local-only option, explicit Hub calls can use
the network. `--local-dir` selects a model directory; in offline mode, its
required checkpoint files must already exist there.

### Training

The private `akiki-training` project supplies Ariadne training for decision models.
Akiki 0.2.1 includes its public API wrapper. The trainer implementation remains private.
Training requires authorized access to a separate local companion installation.
Ariadne adapts the query's representation so a frozen model can use its existing
latent capabilities. The input affine is separate from the classification head.
The general technique applies to general-purpose LLMs and typically trains
fewer than 1% of the base model's parameters. Supported checkpoints depend on
the installed implementation; native inference here supports MiniLM/BERT exports.
Dataset tools and Ariadne Lab have separate companion packages.
Akiki includes the training API wrapper. Trainer implementations and Lab assets
stay in the private companion projects. Datasets and experiment outputs stay local.
Supported MiniLM/BERT specialist exports run through `DecisionModel` without
the training package. Use `MiniLM` for direct access to a compatible export.

Install the optional runtime dependencies through Akiki:

```bash
pip install 'akiki[training]'
```

This extra installs PyTorch, Transformers, and safetensors.
It does not install, download, or include the private training implementation.
The extra alone cannot train a model.

Obtain an authorized local companion checkout or wheel.
For an existing checkout, install its Ariadne dependencies:

```bash
python -m pip install -e 'path/to/akiki-training[ariadne]'
```

The companion is not distributed on PyPI with this release.
Its own dependency requirements also apply. No private code is fetched automatically.

The thin `akiki.training` module forwards the public API declared by
`akiki_training.__all__`, including its intent-training interface:

```python
from akiki import training

result = training.train_intents(
    intents=["question", "task"],
    examples={
        "What time is it?": "question",
        "Where is the station?": "question",
        "Restart the server.": "task",
        "Create a backup.": "task",
    },
    base_model="path/to/local/MiniLM",
    output_dir="models/my-classifier",
)
```

`dir(training)` and wildcard imports follow the companion's current public
exports. Existing named imports such as
`from akiki.training import train_classifier` still work through its original
compatibility API. After editing an imported companion module, restart the
notebook kernel or reload that module; editable installs do not reload Python
objects automatically.

Training dependencies do not load during ordinary inference imports.
An authorized companion installation supplies `akiki-training train` for command-line training.
Ariadne Lab also requires its separate private installation.
Its command is `akiki-lab --open`. Lab runs on your own computer or infrastructure.

## Development

See the [Python style guide](docs/style.md) for enum conventions, development
setup, and lint commands. Ruff checks Python formatting and imports; commit
hooks and CI also run YAML linting.

## License

The package and bundled Akiki model weights are licensed under Apache-2.0.
