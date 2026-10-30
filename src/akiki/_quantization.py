"""Convert verified local FP32 tensors to the native AKVAD003 model format.

Conversion uses NumPy only. The original model is never modified. The binary
layout and per-row quantization match akiki-vad-engine's offline exporter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._precision import ModelPrecision

# AKVAD003 format IDs are fixed independently of enum declaration order.
_PRECISION_IDS = {
    ModelPrecision.FP32: 0,
    ModelPrecision.FP16: 1,
    ModelPrecision.BF16: 2,
    ModelPrecision.INT8: 3,
    ModelPrecision.INT4: 4,
    ModelPrecision.INT8_FP32_INPUT: 5,
}
CONVERTER_VERSION = 1


def _fp32_tensors(data):
    """Read only the bounded, dense F32 subset used by the verified source."""
    if len(data) < 8:
        raise ValueError("Truncated safetensors file")
    header_size = struct.unpack_from("<Q", data)[0]
    if not 2 <= header_size <= min(len(data) - 8, 16 * 1024 * 1024):
        raise ValueError("Invalid safetensors header size")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate safetensors metadata key")
            result[key] = value
        return result

    header = json.loads(data[8 : 8 + header_size], object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    payload = memoryview(data)[8 + header_size :]
    tensors, ranges = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict) or item.get("dtype") != "F32":
            raise ValueError("Quantization requires original FP32 source tensors")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(v) is not int or v <= 0 for v in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            raise ValueError(f"Invalid safetensors shape or offsets: {name}")
        start, end = offsets
        if not 0 <= start <= end <= len(payload) or end - start != math.prod(shape) * 4:
            raise ValueError(f"Invalid safetensors data range: {name}")
        tensor = np.frombuffer(payload[start:end], dtype="<f4").reshape(shape)
        if not np.isfinite(tensor).all():
            raise ValueError(f"Nonfinite source tensor: {name}")
        tensors[name] = tensor
        ranges.append((start, end))
    position = 0
    for start, end in sorted(ranges):
        if start != position:
            raise ValueError("Overlapping or incomplete safetensors payload")
        position = end
    if position != len(payload):
        raise ValueError("Unreferenced safetensors data")
    return tensors


def _matrix(weights, bias, precision: ModelPrecision):
    rows, cols = weights.shape
    stride = (cols + 63) // 64 * 64
    weights = np.pad(weights, ((0, 0), (0, stride - cols)))
    data = bytearray(struct.pack("<III", rows, cols, stride))
    data += np.asarray(bias, dtype="<f4").tobytes()
    if precision is ModelPrecision.FP32:
        packed = weights.astype("<f4")
    elif precision is ModelPrecision.FP16:
        with np.errstate(over="ignore"):
            packed = weights.astype("<f2")
        if not np.isfinite(packed).all():
            raise ValueError("Source weights exceed the finite FP16 range")
    elif precision is ModelPrecision.BF16:
        bits = weights.astype("<f4").view("<u4")
        packed = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
        if np.any((packed & 0x7FFF) == 0x7F80):
            raise ValueError("Source weights exceed the finite BF16 range")
    else:
        limit = 7 if precision is ModelPrecision.INT4 else 127
        scales = np.maximum(np.max(np.abs(weights), axis=1) / limit, 1e-12).astype("<f4")
        quantized = np.clip(np.rint(weights / scales[:, None]), -limit, limit).astype(np.int8)
        data += scales.tobytes()
        data += quantized.astype(np.int32).sum(axis=1).astype("<i4").tobytes()
        packed = (
            ((quantized[:, 0::2] & 15) | ((quantized[:, 1::2] & 15) << 4)).astype(np.uint8)
            if precision is ModelPrecision.INT4
            else quantized
        )
    data += packed.tobytes()
    return data


def convert_model(config, tensors, precision: ModelPrecision):
    """The native loader validates the FP32 source before this conversion."""
    if not isinstance(precision, ModelPrecision):
        raise TypeError("precision must be a ModelPrecision member")
    f = config["frontend"]
    convolutions = config["convolutions"]
    header = [
        _PRECISION_IDS[precision],
        config["hop_samples"],
        config["context_samples"],
        f["fft_size"],
        f["window_size"],
        f["stft_hop"],
        f["right_padding"],
        ["reflect", "zero"].index(f["padding_mode"]),
        config["hidden_size"],
        ["single", "mean", "last", "flatten"].index(config["reduction"]),
        len(convolutions),
    ]
    data = bytearray(b"AKVAD003" + struct.pack("<11I", *header))
    frames = (
        config["hop_samples"] + config["context_samples"] + f["right_padding"] - f["window_size"]
    ) // f["stft_hop"] + 1
    for i, c in enumerate(convolutions):
        w = tensors[f"encoder.{i}.weight"]
        channels, taps = w.shape[1:]
        stride, dilation = c["stride"], c["dilation"]
        left, right = c["padding"]
        out = (frames + left + right - (taps - 1) * dilation - 1) // stride + 1
        # Prune unused padding taps before choosing per-row quantization scales.
        used = []
        for tap in range(taps):
            first = max(0, (left - tap * dilation + stride - 1) // stride)
            if first < out and first * stride + tap * dilation < left + frames:
                used.append(tap)
        if used and used[0] * dilation <= left and (used[0] != 0 or used[-1] != taps - 1):
            w = w[:, :, used[0] : used[-1] + 1]
            left -= used[0] * dilation
            right = max(0, (out - 1) * stride + (w.shape[2] - 1) * dilation + 1 - frames - left)
        data += struct.pack("<6I", channels, w.shape[2], stride, dilation, left, right)
        data += _matrix(
            w.transpose(0, 2, 1).reshape(w.shape[0], -1), tensors[f"encoder.{i}.bias"], precision
        )
        frames = out
    for name in ("ih", "hh"):
        data += _matrix(tensors[f"rnn.weight_{name}_l0"], tensors[f"rnn.bias_{name}_l0"], precision)
    data += tensors["head.weight"].astype("<f4").tobytes()
    data += tensors["head.bias"].astype("<f4").tobytes()
    return bytes(data)


def default_cache_dir():
    if os.environ.get("AKIKI_CACHE_DIR"):
        return Path(os.environ["AKIKI_CACHE_DIR"]).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "akiki/Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/akiki"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "akiki"


def _atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached_model(folder, receipt, precision: ModelPrecision, cache_dir, validate_source):
    """Return a checked variant; repair incomplete/corrupt derived cache entries."""
    identity = {
        "converter_version": CONVERTER_VERSION,
        "model_precision": precision.value,
        "config_sha256": receipt["config_sha256"],
        "source_sha256": receipt["model_safetensors_sha256"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = (
        (Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir())
        / "models"
        / key
    )
    model, manifest = directory / "model.akv", directory / "manifest.json"
    try:
        saved = json.loads(manifest.read_text())
        payload = model.read_bytes()
        if (
            saved["source"] == identity
            and saved["bytes"] == len(payload)
            and saved["sha256"] == hashlib.sha256(payload).hexdigest()
            and payload[:12] == b"AKVAD003" + struct.pack("<I", _PRECISION_IDS[precision])
        ):
            return model
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Check model geometry and the optional Fourier basis with the same native
    # loader used for FP32 inference before producing a different representation.
    validate_source()
    config_bytes = (folder / "config.json").read_bytes()
    weights = (folder / "model.safetensors").read_bytes()
    if (
        hashlib.sha256(config_bytes).hexdigest() != identity["config_sha256"]
        or hashlib.sha256(weights).hexdigest() != identity["source_sha256"]
    ):
        raise ValueError("Source model changed during quantization")
    payload = convert_model(json.loads(config_bytes), _fp32_tensors(weights), precision)
    saved = {
        "source": identity,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(model, payload)
        _atomic_write(manifest, (json.dumps(saved, sort_keys=True, indent=2) + "\n").encode())
    except OSError as error:
        raise RuntimeError(
            "Cannot write the Akiki model cache; set cache_dir or AKIKI_CACHE_DIR to a writable directory"
        ) from error
    return model
