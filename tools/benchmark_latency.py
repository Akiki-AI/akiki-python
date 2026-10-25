"""Measure local VAD call boundaries at an explicitly selected model precision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


CASES = ("auto_push", "rust_push", "onnx_push", "rust_adapter", "rust_ffi", "rust_sequence_64")


def worker(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import ctypes
    import gc
    import hashlib
    import platform
    import time

    import numpy as np
    from akiki import VAD, __version__
    import akiki

    engine = "auto" if args.worker == "auto_push" else "onnx" if args.worker == "onnx_push" else "rust"
    try:
        vad = VAD(engine=engine, library_path=args.library_path,
                  model_precision=args.model_precision, cache_dir=args.cache_dir)
    except (RuntimeError, ValueError) as error:
        return {"case": args.worker, "unavailable": str(error)}
    with vad:
        hop = vad.chunk_samples
        if args.audio:
            import soundfile as sf
            audio, rate = sf.read(args.audio, dtype="float32")
            if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError("Benchmark audio must be finite mono 16 kHz")
        else:
            audio = np.random.default_rng(1729).normal(0, .02, 1024 * hop).astype(np.float32)
        audio = np.ascontiguousarray(audio[:len(audio) // hop * hop])
        if len(audio) < 64 * hop:
            raise ValueError("Benchmark audio needs at least 64 complete hops")
        chunks = list(audio.reshape(-1, hop))
        backend = vad._backend
        per_call = 1
        if args.worker.endswith("_push"):
            call = lambda i: vad.push(chunks[i % len(chunks)])
            scope = "Public push: validation, buffering, frontend, GRU, state handling, timestamps and SpeechFrame"
        elif args.worker == "rust_adapter":
            padded = np.pad(audio, (vad.context_samples, 0))
            windows = [padded[i * hop:i * hop + vad.context_samples + hop] for i in range(len(chunks))]
            call = lambda i: backend.infer(windows[i % len(windows)], vad.state)
            scope = "Rust Python adapter: native inference, pointer conversion, state allocation and readback"
        elif args.worker == "rust_ffi":
            pointers = [chunk.ctypes.data for chunk in chunks]
            function, handle = backend.lib.akiki_vad_chunk_len, backend.handle
            call = lambda i: function(handle, pointers[i % len(pointers)], hop)
            scope = "One checked native hop through ctypes; precomputed pointers; no Python buffering or state readback"
        else:
            per_call = 64
            function, handle = backend.lib.akiki_vad_sequence, backend.handle
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            output = np.empty(per_call, np.float32)
            pointers = [audio[i:i + per_call * hop].ctypes.data
                        for i in range(0, len(audio) - per_call * hop + 1, per_call * hop)]
            output_pointer = output.ctypes.data
            call = lambda i: function(handle, pointers[i % len(pointers)], per_call, output_pointer)
            scope = "64 consecutive recurrent hops in native code; amortized us/hop, not one-call latency"
        warmup = max(1, args.warmup // per_call)
        iterations = max(100, args.iterations // per_call)
        for i in range(warmup):
            call(i)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            elapsed = []
            for i in range(iterations):
                start = time.perf_counter_ns()
                last = call(i)
                elapsed.append((time.perf_counter_ns() - start) / 1000 / per_call)
            start = time.perf_counter_ns()
            for i in range(iterations):
                call(i)
            amortized = (time.perf_counter_ns() - start) / iterations / 1000 / per_call
        finally:
            if was_enabled:
                gc.enable()
        probability = (last[0].probability if args.worker.endswith("_push") else
                       last[0] if args.worker == "rust_adapter" else
                       last if args.worker == "rust_ffi" else output[-1])
        assert np.isfinite(probability) and 0 <= probability <= 1
        if per_call > 1:
            assert last == 0 and np.isfinite(output).all()
        library = getattr(backend, "library_path", None)
        return dict(case=args.worker, engine=vad.engine, scope=scope,
                    median_us=float(np.median(elapsed)), p95_us=float(np.quantile(elapsed, .95)),
                    amortized_us=amortized, iterations=iterations, warmup_calls=warmup,
                    hops_per_call=per_call, sample_rate=16000, hop_samples=hop,
                    python=platform.python_version(), numpy=np.__version__, version=__version__,
                    package_path=akiki.__file__, model_id=vad.model_id,
                    model_precision=vad.model_precision, fft_precision=vad.fft_precision,
                    library_path=str(library) if library else None,
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library else None,
                    engine_errors=vad.engine_errors,
                    cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="Existing local mono 16 kHz audio; otherwise use seeded noise")
    parser.add_argument("--library-path", type=Path)
    parser.add_argument("--model-precision", default=None,
                        help="Default: INT8 for Rust, FP32 for ONNX/PyTorch",
                        choices=["fp32", "fp16", "bf16", "int8", "int4", "int8-fp32-input"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cpu", type=int, help="Pin each worker to this logical CPU (Linux)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("Positive rounds/iterations and nonnegative warmup are required")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    runs = []
    for round_index in range(args.rounds):
        for case in CASES[round_index:] + CASES[:round_index]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                       "--iterations", str(args.iterations), "--warmup", str(args.warmup)]
            for flag, value in [("--audio", args.audio), ("--cpu", args.cpu), ("--library-path", args.library_path),
                                ("--model-precision", args.model_precision), ("--cache-dir", args.cache_dir)]:
                if value is not None:
                    command.extend([flag, str(value)])
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            run = {**json.loads(result.stdout), "round": round_index + 1}
            runs.append(run)
            print(f"round {round_index + 1} {case}: " +
                  (f"{run['median_us']:.3f} us/hop" if "median_us" in run else run['unavailable']), flush=True)
    summary = {}
    for case in CASES:
        group = [r for r in runs if r["case"] == case and "median_us" in r]
        if group:
            summary[case] = {key: statistics.median(r[key] for r in group)
                             for key in ["median_us", "p95_us", "amortized_us"]}
            summary[case].update(scope=group[0]["scope"], engine=group[0]["engine"],
                                 round_median_range_us=[min(r["median_us"] for r in group),
                                                        max(r["median_us"] for r in group)])
    receipt = dict(audio=str(args.audio) if args.audio else "Seeded float32 noise", rounds=args.rounds,
                   method="Fresh process per case/round; warmed; median of round medians; GC disabled while timing",
                   model_precision=args.model_precision or "engine default", fft_precision="fp32",
                   excludes="Model conversion/loading, audio decoding, and waiting to receive a 16 ms audio hop",
                   model_downloads=0, summary=summary, runs=runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
