# Akiki

Akiki is a lightweight Python package for streaming voice activity detection.
It identifies speech in audio and returns a speech probability, a speech/silence
decision, and timestamps for each 16 ms frame.

The model is included in the package. Processing runs locally, without a model
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
        print(frame.start_seconds, frame.end_seconds,
              frame.probability, frame.speech)
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

help(akiki.VAD)              # Configuration, attributes, and streaming example
help(akiki.VAD.__init__)     # Constructor parameters and defaults
help(akiki.VAD.push)         # Audio input, buffering, results, and errors
help(akiki.SpeechFrame)      # Result fields and timestamp units
```

## Command line

```bash
akiki recording.wav
akiki recording.wav --threshold 0.6
```

The command prints one JSON object per complete frame, containing its timestamps,
speech probability, and speech decision.

## Inference engines

Linux x86_64 wheels include the compiled Rust engine for systems with glibc
2.34 or later. Pip and uv select the compatible wheel automatically; no Rust
toolchain is needed to install it.

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

Choose an engine with `VAD(engine="rust")`, `VAD(engine="onnx")`, or
`VAD(engine="pytorch")`. The command-line equivalent is `--engine rust`,
`--engine onnx`, or `--engine pytorch`. An explicitly selected engine raises
an error if it is unavailable. Check `vad.engine` to see the active engine.

All engines use the same bundled model and streaming API.

## Model precision

`VAD()` and the command line use INT8 by default when running the Rust engine.
The package generates the INT8 weights from the bundled FP32 model on first use
and reuses them from the local cache. The FFT remains FP32.

Choose another precision when creating the detector. For example, to use the
original FP32 weights:

```python
with VAD(engine="rust", model_precision="fp32") as vad:
    frames = vad.push(audio, sample_rate=sample_rate)
```

The command-line equivalent is:

```bash
akiki recording.wav --model-precision fp32
```

| `model_precision` | Convolution and GRU projections |
| --- | --- |
| `int8` (Rust default) | INT8 weights and dynamically quantized activations |
| `fp32` | Original FP32 weights and activations |
| `fp16` | FP16 weights, FP32 activations and accumulation |
| `bf16` | BF16 weights and projection activations, FP32 accumulation |
| `int4` | Packed INT4 weights and dynamically quantized INT8 activations |
| `int8-fp32-input` | INT8 weights; first projection keeps FP32 activations |

The FFT stays FP32: `fft_precision` accepts only `"fp32"`. Recurrent state,
nonlinearities, biases, and the output head also remain FP32. Lower precision
can change speech probabilities; speed depends on the CPU and chosen format.

Conversion runs locally using the included weights, without downloading models
or installing PyTorch or ONNX Runtime. Generated variants are reused from the
user cache. Set `cache_dir` or `AKIKI_CACHE_DIR` to choose another location. The
original weights are preserved; incomplete or corrupt cache entries are rebuilt.

Check `vad.model_precision` and `vad.fft_precision` to inspect the settings.
When no model precision is specified, ONNX and PyTorch use FP32, including when
`engine="auto"` falls back to either runtime. Explicit precision requests are
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

## License

The package and bundled Akiki model weights are licensed under Apache-2.0.
