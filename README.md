# Akiki

Akiki is a lightweight Python package for streaming voice activity detection.
It identifies speech in audio and returns a speech probability, a speech/silence
decision, and timestamps for each 16 ms frame. It can also run local MiniLM text
models for sentence embeddings and classification; see
[MiniLM text models](#minilm-text-models).

The VAD model is included in the package. Processing runs locally, without a model
download, network connection, or external service. The Rust engine uses INT8
model inference by default, with an FP32 FFT.

## Install

Requires Python 3.10 or later.

```bash
pip install akiki
```

Or add it to a uv project:

```bash
uv add akiki
```

The base installation uses the bundled Rust engine on supported hosts. On other
platforms, install an alternate engine with `pip install 'akiki[onnx]'` or
`pip install 'akiki[pytorch]'`.

## Python usage

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

help(akiki.VAD)  # Configuration, attributes, and streaming example
help(akiki.VAD.__init__)  # Constructor parameters and defaults
help(akiki.VAD.push)  # Audio input, buffering, results, and errors
help(akiki.SpeechFrame)  # Result fields and timestamp units
```

## Command line

```bash
akiki vad recording.wav
akiki vad recording.wav --threshold 0.6
akiki vad --help
```

The command prints one JSON object per complete frame, containing its timestamps,
speech probability, and speech decision.

VAD options follow the `vad` subcommand. Use `akiki --help` to list available
commands. The equivalent module invocation is `python -m akiki vad recording.wav`.

## Inference engines

Linux x86_64 wheels include the compiled Rust engine for systems with glibc
2.34 or later. Pip and uv select the compatible wheel automatically; no Rust
toolchain is needed to install it.

Version **0.1.2** includes **Linux ARM64 VAD support**, including Raspberry Pi 5
with a 64-bit OS and glibc 2.28 or later. This wheel includes the Rust VAD library
and supports all six VAD precision modes with the same INT8 default. INT8
projections automatically use KleidiAI Neon DotProd kernels on compatible CPUs,
with a portable fallback on older ARM64 CPUs. Other precisions retain portable
execution. The optimized path has been tested on a physical Raspberry Pi 5.
MiniLM remains unsupported in the ARM64 wheel.

Akiki selects the Rust engine when a compatible native library is available,
then tries ONNX Runtime and PyTorch if they are installed. Both are optional;
the base package requires neither runtime.

To use ONNX Runtime:

```bash
pip install 'akiki[onnx]'
# or: uv add 'akiki[onnx]'
```

To use PyTorch:

```bash
pip install 'akiki[pytorch]'
# or: uv add 'akiki[pytorch]'
```

The PyTorch extra does not install ONNX Runtime. Install `akiki[onnx,pytorch]`
if you want both alternatives. If no engine is available, Akiki raises an error
with installation instructions.

Import `Engine` from `akiki` and choose an engine with `VAD(engine=Engine.RUST)`,
`VAD(engine=Engine.ONNX)`, or `VAD(engine=Engine.PYTORCH)`. The default is
`Engine.AUTO`; the Python API requires enum members. On the command line, use
`akiki vad recording.wav --engine rust`, `--engine onnx`, or `--engine pytorch`.
An explicitly selected engine raises an error if it is unavailable. Check
`vad.engine` to see the active `Engine` member. `vad.engine_errors` maps
`Engine` members to the reasons they were unavailable during automatic selection.

All engines use the same bundled model and streaming API.

## Model precision

`VAD()` and the command line use INT8 by default when running the Rust engine.
The package generates the INT8 weights from the bundled FP32 model on first use
and reuses them from the local cache. The FFT remains FP32.

Choose another precision when creating the detector. For example, to use the
original FP32 weights:

```python
from akiki import Engine, ModelPrecision, VAD

with VAD(engine=Engine.RUST, model_precision=ModelPrecision.FP32) as vad:
    frames = vad.push(audio, sample_rate=sample_rate)
```

The command-line equivalent is:

```bash
akiki vad recording.wav --model-precision fp32
```

| `model_precision` | Convolution and GRU projections |
| --- | --- |
| `ModelPrecision.INT8` (Rust default) | INT8 weights and dynamically quantized activations |
| `ModelPrecision.FP32` | Original FP32 weights and activations |
| `ModelPrecision.FP16` | FP16 weights, FP32 activations and accumulation |
| `ModelPrecision.BF16` | BF16 weights and projection activations, FP32 accumulation |
| `ModelPrecision.INT4` | Packed INT4 weights and dynamically quantized INT8 activations |
| `ModelPrecision.INT8_FP32_INPUT` | INT8 weights; first projection keeps FP32 activations |

The Python VAD API requires `ModelPrecision` and `FFTPrecision` enum members,
both exported from `akiki`. Omit `model_precision` or pass `None` to use the
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
always honored: non-FP32 precision requires Rust and raises an error if a
compatible Rust engine is unavailable.

## Precision and performance

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

## MiniLM text models

`akiki.MiniLM` runs MiniLM sentence encoders locally with the Rust engine. It
loads plain checkpoints, such as `sentence-transformers/all-MiniLM-L6-v2`, and
Akiki Ariadne variants that add a small adapter to the frozen encoder, such as
the Akiki MiniLM-L6-v2 Clover intent classifier.

The MiniLM weights are not included in the package. Pass the path to a local
copy when creating the model:

```python
from akiki import MiniLM

with MiniLM("path/to/all-MiniLM-L6-v2") as model:
    vectors = model.embed(["A man is eating food.", "Someone is eating a meal."])
    print(vectors[0] @ vectors[1])  # cosine similarity, about 0.78
```

Embeddings are float32 NumPy arrays, mean-pooled and L2-normalized, so their dot
products are cosine similarities. `embed()` accepts one string or a list.

Variants with a head classify text in their own input format:

```python
with MiniLM("path/to/Akiki-MiniLM-L6-v2-clover-intent") as model:
    result = model.classify("Current user message: Restart my web server.")
    print(result.label, result.probabilities[result.index])  # task, about 0.92

    follow_up = model.classify(
        "Conversation context:\n"
        "assistant: Would you like me to check whether last night's backup ran?\n\n"
        "Current user message: yes"
    )
    print(follow_up.label)  # question
```

Each classification has the most probable output's `index` and `label`, the
`probabilities` of every output, and the raw `logits`. Probabilities are a plain
softmax, without calibration.

A model folder is either a checkpoint with `config.json`, `model.safetensors`,
and `tokenizer.json`, or a variant folder holding that checkpoint in
`base_model/` beside `adapter.safetensors`. A variant's optional
`model_config.json` names its outputs (`labels`) and sets a chunk length for long
text (`max_length`). Pass `adapter_path` to use an adapter stored elsewhere, or a
variant's `base_model` folder to use the plain encoder.

Each sequence may contain up to 512 tokens, including special tokens. Longer
text raises `ValueError` rather than being truncated, unless `chunk_tokens` is
set. Chunked text is split into sequences of at most that many tokens, and their
embeddings are combined by a mean weighted by their text tokens. Retrieval
variants trained this way set `max_length`, so chunking is automatic.
sentence-transformers truncates all-MiniLM-L6-v2 input after 256 tokens instead,
so its embeddings of longer text differ.

For task-specific inputs, such as typed decisions, `tokenize()` and
`encode_chunks()` give direct control over the chunks and their weights;
`encode()` returns both the embedding and any head logits. One model can be
shared between threads; its calls run one at a time.

MiniLM requires the Rust engine included in Linux x86_64 wheels; it has no ONNX
or PyTorch alternative. It uses up to four CPU threads by default; set
`threads` to change this. Projections use INT8 by default. Choose
`model_precision="fp32"` to use the original weights, or `fp16`, `bf16`,
`int8-fp32-input`, `int4`, or `int4-g32`. Weights are converted in memory when
the model loads; the files are unchanged. Attention, normalization, and pooling
remain FP32. Lower precision uses less memory but can change embeddings,
probabilities and, for borderline inputs, the label.

On the command line:

```bash
akiki minilm --model path/to/all-MiniLM-L6-v2 "A man is eating food."
akiki minilm --model path/to/Akiki-MiniLM-L6-v2-clover-intent \
    "Current user message: Restart my web server."
akiki minilm --help
```

The command prints JSON: the classification for models with a head, and the
embedding otherwise. Add `--embedding` to include the embedding as well.

### Hugging Face models

Install the optional Hub integration to explicitly download compatible MiniLM
checkpoints or Akiki Ariadne variants:

```bash
pip install 'akiki[huggingface]'
```

```python
from akiki import MiniLM, download_model

with MiniLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") as model:
    vector = model.embed("A man is eating food.")

# Fetch a model without loading an inference engine, then use the local path.
path = download_model("sentence-transformers/all-MiniLM-L6-v2")
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
# Fetch and run a model.
akiki minilm --repo sentence-transformers/all-MiniLM-L6-v2 "A man is eating food."

# Fetch only, writing into a chosen folder. The command prints its local path.
akiki download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/minilm

# Reuse a previously cached model without network access.
akiki minilm --repo sentence-transformers/all-MiniLM-L6-v2 --local-files-only "Hello."
```

Downloads use the [Hugging Face Hub cache and revision handling](https://huggingface.co/docs/huggingface_hub/guides/download).
Private or gated repositories use `HF_TOKEN` or your saved Hugging Face login;
the Python API also accepts `token`. Set `HF_HUB_CACHE` or pass `cache_dir` to
choose a cache location. `local_files_only=True` or `--local-files-only` requires
an existing local snapshot; missing required files produce an error.
`HF_HUB_OFFLINE=1` also forces local-only access and is never overridden.
Ordinary `MiniLM(path)` loading and inference do not contact Hugging Face.

## Development

See the [Python style guide](docs/style.md) for enum conventions, development
setup, and lint commands. Ruff checks Python formatting and imports; commit
hooks and CI also run YAML linting.

## License

The package and bundled Akiki model weights are licensed under Apache-2.0.
