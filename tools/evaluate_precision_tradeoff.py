"""Evaluate installed Akiki precisions on explicitly supplied, local frozen data.

Uses the existing 32 ms TEN/VoxConverse protocol: pair-mean 16 ms outputs,
no time shift or smoothing, reset between recordings, pooled exact average
precision. Native sequence inference is checked against public VAD.push().
No models, datasets, or dependencies are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from akiki import __version__
from akiki.audio.vad import VAD, Engine, FFTPrecision, ModelPrecision

MODES = tuple(ModelPrecision)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def labels_for(suite, row, samples):
    if suite == "ten":
        annotation = Path(row["path"]).with_suffix(".scv")
        assert sha256(annotation) == row["annotation_sha256"]
        values = annotation.read_text().strip().split(",")[1:]
        start, end = np.asarray(values[::3], float), np.asarray(values[1::3], float)
        labels = np.asarray(values[2::3], np.uint8)
        # Preserve the benchmark's segment-duration rounding, including drift.
        counts = np.round((end - start) / 0.032).astype(np.int32)
        result = np.repeat(labels, counts)
        return result[: min(len(result), int((end[-1] - start[0]) / 0.032))]
    merged = []
    for segment in sorted(row["speech_ts"], key=lambda x: x["start"]):
        start, end = max(0.0, segment["start"]), min(samples / 16000, segment["end"])
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    count = samples // 512
    occupancy = np.zeros(count, np.float64)
    for start, end in merged:
        start, end = int(start * 16000), int(end * 16000)
        indexes = np.arange(start // 512, min(count, (end + 511) // 512))
        occupancy[indexes] += np.maximum(
            0, np.minimum(end, (indexes + 1) * 512) - np.maximum(start, indexes * 512)
        )
    return (occupancy > 256).astype(np.uint8)


def predict(vad, audio):
    vad.reset()
    count = len(audio) // vad.chunk_samples
    output = np.empty(count, np.float32)
    fn = vad._backend.lib.akiki_vad_sequence
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    # Sequence preserves recurrence; it does not batch independent frames.
    status = fn(vad._backend.handle, audio.ctypes.data, count, output.ctypes.data)
    if status or not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise RuntimeError("Native sequence inference failed")
    return output


def benchmark(args):
    runs = []
    worker = Path(__file__).with_name("benchmark_latency.py")
    for round_index in range(3):
        modes = MODES[round_index:] + MODES[:round_index]
        for mode in modes:
            cases = (
                ("akiki_ffi", "akiki_push") if round_index % 2 == 0 else ("akiki_push", "akiki_ffi")
            )
            for case in cases:
                command = [
                    sys.executable,
                    str(worker),
                    "--worker",
                    case,
                    "--model-precision",
                    mode.value,
                    "--cache-dir",
                    str(args.output / "cache"),
                    "--cpu",
                    str(args.cpu),
                    "--iterations",
                    "7500",
                    "--warmup",
                    "2000",
                    "--audio",
                    str(args.ten_root / "testset/testset-audio-01.wav"),
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                run = json.loads(result.stdout)
                if "unavailable" in run:
                    raise RuntimeError(run["unavailable"])
                runs.append({**run, "round": round_index + 1})
                print(
                    f"Timing {round_index + 1}/3 {mode} {case}: {run['median_us']:.3f} us",
                    flush=True,
                )
    summary = {}
    for mode in MODES:
        summary[mode.value] = {}
        for case in ("akiki_ffi", "akiki_push"):
            group = [r for r in runs if r["model_precision"] == mode.value and r["case"] == case]
            summary[mode.value][case] = dict(
                median_us=statistics.median(r["median_us"] for r in group),
                round_medians_us=[r["median_us"] for r in group],
                p95_us=statistics.median(r["p95_us"] for r in group),
            )
    receipt = dict(
        summary=summary,
        runs=runs,
        worker_sha256=sha256(worker),
        audio_sha256=sha256(args.ten_root / "testset/testset-audio-01.wav"),
        method="Three fresh processes per mode/case, rotated order; 2000 warmup and 7500 timed calls; median of round medians; GC disabled during timing",
        excludes="Model conversion/loading, audio decoding and the 16 ms needed to receive an audio hop",
        range_note="Min/max of three round medians, not a confidence interval",
    )
    dump(args.output / "latency.json", receipt)
    return receipt


def evaluate(args, suite, vads):
    folder = args.reference / suite
    with np.load(folder / "predictions.npz") as z:
        reference = {key: z[key].copy() for key in ("labels", "lengths", "sample_ids")}
        old_fp32 = z["teacher-only-s1730"].copy()
    with np.load(folder / "native16-predictions.npz") as z:
        np.testing.assert_array_equal(reference["sample_ids"], z["sample_ids"])
        native_lengths = z["lengths"].copy()
    if suite == "ten":
        frozen = {
            r["sample_id"]: r
            for r in json.loads((args.ten_root / "results/metrics.json").read_text())["files"]
        }
        records = [
            dict(
                sample_id=str(identity),
                path=str(args.ten_root / "testset" / (str(identity) + ".wav")),
                sha256=frozen[str(identity)]["audio_sha256"],
                annotation_sha256=frozen[str(identity)]["annotation_sha256"],
            )
            for identity in reference["sample_ids"]
        ]
    else:
        records = [json.loads(line) for line in args.vox_manifest.read_text().splitlines()]
        assert all(r["split"] == "human_test" for r in records)
    np.testing.assert_array_equal([r["sample_id"] for r in records], reference["sample_ids"])
    parts = {mode: [] for mode in MODES}
    parity = {mode: 0.0 for mode in MODES}
    offset = 0
    started = time.perf_counter()
    for index, (row, length, native_length) in enumerate(
        zip(records, reference["lengths"], native_lengths, strict=True)
    ):
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Evaluation audio changed: " + row["sample_id"])
        audio, rate = sf.read(row["path"], dtype="float32")
        assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
        assert len(audio) // 256 == native_length
        np.testing.assert_array_equal(
            labels_for(suite, row, len(audio))[:length],
            reference["labels"][offset : offset + length],
        )
        for mode, vad in vads.items():
            values = predict(vad, audio)
            assert len(values) >= 2 * length
            if index == 0:
                vad.reset()
                actual = np.asarray(
                    [
                        vad.push(chunk)[0].probability
                        for chunk in audio[: 128 * 256].reshape(-1, 256)
                    ],
                    np.float32,
                )
                error = float(np.max(np.abs(actual - values[:128])))
                parity[mode] = error
                np.testing.assert_array_equal(actual, values[:128])
            parts[mode].append(values[: 2 * length].reshape(-1, 2).mean(1))
        offset += int(length)
        print(
            f"Scoring {suite}: {index + 1}/{len(records)} recordings ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    assert offset == len(reference["labels"])
    predictions = {mode: np.concatenate(values) for mode, values in parts.items()}
    # Compare the actual Akiki FP32 computation with the frozen original model.
    fp32_difference = float(np.max(np.abs(predictions[ModelPrecision.FP32] - old_fp32)))
    np.testing.assert_allclose(predictions[ModelPrecision.FP32], old_fp32, rtol=0, atol=2e-4)
    metrics = {}
    for mode, probabilities in predictions.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            reference["labels"], probabilities >= 0.5, average="binary"
        )
        metrics[mode.value] = dict(
            average_precision=float(average_precision_score(reference["labels"], probabilities)),
            precision_at_0_5=float(precision),
            recall_at_0_5=float(recall),
            f1_at_0_5=float(f1),
        )
    np.savez_compressed(
        args.output / (suite + "-predictions.npz"),
        **reference,
        **{mode.value: values for mode, values in predictions.items()},
    )
    receipt = dict(
        models=metrics,
        recordings=len(records),
        frames=offset,
        scored_seconds=offset * 0.032,
        speech_prevalence=float(reference["labels"].mean()),
        public_api_parity_max_abs_error={mode.value: error for mode, error in parity.items()},
        fp32_reference_max_abs_error=fp32_difference,
        reference_predictions_sha256=sha256(folder / "predictions.npz"),
        reference_native16_sha256=sha256(folder / "native16-predictions.npz"),
        audio_sha256={r["sample_id"]: r["sha256"] for r in records},
    )
    dump(args.output / (suite + "-metrics.json"), receipt)
    print(
        suite,
        json.dumps({m: round(v["average_precision"] * 100, 6) for m, v in metrics.items()}),
        flush=True,
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ten-root", type=Path, required=True)
    parser.add_argument("--vox-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=8)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    latency = benchmark(args)
    os.sched_setaffinity(0, {args.cpu})
    vads = {
        mode: VAD(
            engine=Engine.AKIKI,
            model_precision=mode,
            fft_precision=FFTPrecision.FP32,
            cache_dir=args.output / "cache",
        )
        for mode in MODES
    }
    try:
        models = {}
        for mode, vad in vads.items():
            backend = vad._backend
            path = backend.model_path
            models[mode.value] = dict(
                model_id=vad.model_id,
                library_sha256=sha256(backend.library_path),
                model_sha256=sha256(path / "model.safetensors" if path.is_dir() else path),
            )
        for run in latency["runs"]:
            assert run["library_sha256"] == models[run["model_precision"]]["library_sha256"]
            assert run["model_id"] == models[run["model_precision"]]["model_id"]
        evaluations = {suite: evaluate(args, suite, vads) for suite in ("ten", "voxconverse")}
    finally:
        for vad in vads.values():
            vad.close()
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        package_version=__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        cpu=cpu,
        cpu_affinity=[args.cpu],
        model_id=models[ModelPrecision.FP32.value]["model_id"],
        fft_precision=FFTPrecision.FP32.value,
        models=models,
        metric="Exact pooled average precision (step-integral PR area; not trapezoidal AUC)",
        protocol="Mean pairs of native 16 ms probabilities onto the frozen 32 ms labels; no time shifts or smoothing; reset between recordings",
        evaluation_status="Previously inspected development benchmarks; small AP differences are not evidence of significance",
        script_sha256=sha256(__file__),
        model_downloads=0,
        latency=latency["summary"],
        evaluations=evaluations,
    )
    dump(args.output / "results.json", receipt)
    with (args.output / "precision-vs-latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "precision",
                "fft_precision",
                "native_us",
                "python_push_us",
                "ten_ap_pct",
                "voxconverse_ap_pct",
            ],
        )
        writer.writeheader()
        for mode in MODES:
            writer.writerow(
                dict(
                    precision=mode.value,
                    fft_precision=FFTPrecision.FP32.value,
                    native_us=latency["summary"][mode.value]["akiki_ffi"]["median_us"],
                    python_push_us=latency["summary"][mode.value]["akiki_push"]["median_us"],
                    ten_ap_pct=evaluations["ten"]["models"][mode.value]["average_precision"] * 100,
                    voxconverse_ap_pct=evaluations["voxconverse"]["models"][mode.value][
                        "average_precision"
                    ]
                    * 100,
                )
            )


if __name__ == "__main__":
    main()
