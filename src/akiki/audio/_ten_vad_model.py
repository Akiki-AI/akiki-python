"""Prepare TEN-VAD ONNX weights for the native streaming engine."""

import hashlib
import re
from pathlib import Path

import numpy as np

from .._atomic import write_bytes
from .._quantization import default_cache_dir
from ..hub import _download_snapshot

_REPO = "TEN-framework/ten-vad"
_REVISION = "bda8ffc78b1846c5c7cbd38f04e52deff49de707"
_ONNX = "src/onnx_model/ten-vad.onnx"
_COEFFICIENTS = "src/coeff.h"
_HUB_CHECKSUMS = {
    _ONNX: "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4",
    _COEFFICIENTS: "0c86dc4a9161e418da6ce03d50dea067dff085749d730e3913b99a8e116397ef",
}
_GRAPH_SHA256 = "bbefc914a8b2a8db673e9b74914cc086b566c09e7a67c22a1f21b26a39edf91a"
_CONVERTER_VERSION = b"ten-vad-onnx-v1"
_NATIVE_MAGIC = b"AKTNV001"
_NATIVE_SIZE = len(_NATIVE_MAGIC) + (75498 + 850) * 4
_PREFIX = "StatefulPartitionedCall/vad_model/"


def _read_sources(model_path: str | Path | None) -> tuple[bytes, bytes]:
    if model_path is None:
        try:
            snapshot = _download_snapshot(
                _REPO,
                revision=_REVISION,
                allow_patterns=tuple(_HUB_CHECKSUMS),
                local_files_only=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Cannot load TEN-VAD from the local Hugging Face cache. "
                f"Pass model_path to a local TEN-VAD ONNX file. {error}"
            ) from error
        contents = []
        for filename, checksum in _HUB_CHECKSUMS.items():
            path = snapshot / filename
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"TEN-VAD Hub file has an incorrect checksum: {path}")
            contents.append(data)
        return contents[0], contents[1]

    weights = Path(model_path).expanduser()
    if weights.suffix.lower() != ".onnx":
        raise ValueError("TEN-VAD model_path must point to its .onnx weights")
    model_data = weights.read_bytes()
    source_dir = weights.parent
    if source_dir.name == "onnx_model" and not (source_dir / "coeff.h").is_file():
        source_dir = source_dir.parent
    coefficients = source_dir / "coeff.h"
    if not coefficients.is_file():
        raise FileNotFoundError(
            "TEN-VAD requires the matching upstream coeff.h beside the ONNX file "
            f"or in the parent of its onnx_model directory: {coefficients}"
        )
    return model_data, coefficients.read_bytes()


def _network_weights(tensors: dict[str, np.ndarray]) -> list[np.ndarray]:
    arrays = []
    used = set()

    def take(name, shape):
        value = tensors.get(name)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid or missing TEN-VAD tensor: {name}")
        used.add(name)
        return value

    for layer, depth, depth_shape, point, point_shape in (
        (
            "separable_conv2d",
            "const_fold_opt__178",
            (1, 1, 3, 3),
            "separable_conv2d/ReadVariableOp_1:0",
            (16, 1, 1, 1),
        ),
        (
            "separable_conv1d",
            "const_fold_opt__179",
            (16, 1, 1, 3),
            "ExpandDims_2:0",
            (16, 16, 1, 1),
        ),
        (
            "separable_conv1d_1",
            "const_fold_opt__180",
            (16, 1, 1, 3),
            "ExpandDims_2:0",
            (16, 16, 1, 1),
        ),
    ):
        arrays.extend(
            (
                take(depth, depth_shape),
                take(f"{_PREFIX}{layer}/{point}", point_shape),
                take(f"{_PREFIX}{layer}/BiasAdd/ReadVariableOp:0", (16,)),
            )
        )

    # ONNX uses IOFC gate order. The native engine uses IFCO.
    gates = [0, 2, 3, 1]
    for input_name, recurrent_name, bias_name, inputs in (
        ("W0__70", "R0__71", "B0__72", 80),
        ("W0__99", "R0__100", "B0__101", 64),
    ):
        weight = take(input_name, (1, 256, inputs)).reshape(4, 64, inputs)[gates]
        recurrent = take(recurrent_name, (1, 256, 64)).reshape(4, 64, 64)[gates]
        bias = take(bias_name, (1, 512)).reshape(2, 4, 64)[:, gates]
        arrays.extend((weight, bias[0], recurrent, bias[1]))

    for layer, inputs, outputs in (("dense_3", 128, 32), ("dense_5", 32, 1)):
        arrays.append(take(f"{_PREFIX}{layer}/Tensordot/ReadVariableOp:0", (inputs, outputs)).T)
        arrays.append(take(f"{_PREFIX}{layer}/BiasAdd/ReadVariableOp:0", (outputs,)))
    if used != tensors.keys():
        raise ValueError("Unsupported TEN-VAD weight set")
    return arrays


def _frontend_weights(data: bytes) -> list[np.ndarray]:
    text = data.decode("utf-8")
    arrays = []
    for name, length in (
        ("AUP_AED_FEATURE_MEANS", 41),
        ("AUP_AED_FEATURE_STDS", 41),
        ("AUP_AED_STFTWindow_Hann768", 768),
    ):
        match = re.search(rf"\b{name}\[[^\]]+\]\s*=\s*\{{([^}}]+)\}}", text)
        if match is None:
            raise ValueError(f"Missing TEN-VAD frontend coefficient array: {name}")
        values = np.array(
            [float(value.strip().rstrip("f")) for value in match[1].split(",") if value.strip()],
            dtype="<f4",
        )
        if values.size != length or not np.isfinite(values).all():
            raise ValueError(f"Invalid TEN-VAD frontend coefficient array: {name}")
        arrays.append(values)
    _, deviations, window = arrays
    if (deviations <= 0).any() or ((window < 0) | (window > 1)).any():
        raise ValueError("Invalid TEN-VAD frontend normalization or window")
    return arrays


def _convert(model_data: bytes, coefficients: bytes) -> bytes:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise RuntimeError("TEN-VAD ONNX import requires pip install 'akiki[ten-vad]'") from error

    try:
        graph = onnx.load_model_from_string(model_data).graph
    except Exception as error:
        raise ValueError("Cannot parse the TEN-VAD ONNX checkpoint") from error

    topology = onnx.GraphProto()
    topology.CopyFrom(graph)
    weights = {}
    for initializer, tensor in zip(graph.initializer, topology.initializer):
        if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("TEN-VAD ONNX must contain its weights in a single file")
        if initializer.data_type == onnx.TensorProto.FLOAT:
            if initializer.name in weights:
                raise ValueError(f"Duplicate TEN-VAD tensor: {initializer.name}")
            weights[initializer.name] = numpy_helper.to_array(initializer)
            for field in ("raw_data", "float_data", "double_data"):
                tensor.ClearField(field)
    if hashlib.sha256(topology.SerializeToString()).hexdigest() != _GRAPH_SHA256:
        raise ValueError("Unsupported ONNX graph. Use the TEN-framework TEN-VAD 1.0 model.")

    arrays = _network_weights(weights) + _frontend_weights(coefficients)
    return _NATIVE_MAGIC + b"".join(array.astype("<f4", copy=False).tobytes() for array in arrays)


def prepare_model(model_path: str | Path | None, cache_dir: str | Path | None) -> Path:
    """Return a checked native cache entry for the selected TEN-VAD ONNX files."""
    weights, coefficients = _read_sources(model_path)
    key = hashlib.sha256(
        _CONVERTER_VERSION
        + hashlib.sha256(weights).digest()
        + hashlib.sha256(coefficients).digest()
    ).hexdigest()
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    directory = root / "ten-vad" / key
    model = directory / "model.bin"
    checksum = directory / "sha256"
    try:
        cached = model.read_bytes()
        if (
            len(cached) == _NATIVE_SIZE
            and cached.startswith(_NATIVE_MAGIC)
            and hashlib.sha256(cached).hexdigest().encode() == checksum.read_bytes()
        ):
            return model
    except OSError:
        pass

    payload = _convert(weights, coefficients)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_bytes(model, payload)
        write_bytes(checksum, hashlib.sha256(payload).hexdigest().encode())
    except OSError as error:
        raise RuntimeError(
            "Cannot write the TEN-VAD model cache. Set cache_dir or AKIKI_CACHE_DIR."
        ) from error
    return model
