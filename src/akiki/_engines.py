"""Lazy local inference adapters. Engine selection never builds or downloads."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import weakref
from enum import Enum
from pathlib import Path

import numpy as np

from ._precision import ModelPrecision


class Engine(Enum):
    """VAD engine choice, with Akiki as the primary engine.

    ``AUTO`` tries ``AKIKI`` first. It then tries the ``ONNX`` and ``PYTORCH``
    compatibility fallbacks. An explicit choice uses only that engine.
    """

    AUTO = "auto"
    AKIKI = "akiki"
    ONNX = "onnx"
    PYTORCH = "pytorch"

    def __str__(self) -> str:
        return self.value


class EngineUnavailable(RuntimeError):
    """An engine or its local model format is unavailable on this host."""


def verify_file(path, expected):
    if not path.is_file():
        raise EngineUnavailable(f"Local model file is missing: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model checksum does not match its metadata: {path}")


class Backend:
    closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("Detector is closed")

    def reset(self):
        self.check_open()

    def close(self):
        self.closed = True


class OnnxBackend(Backend):
    def __init__(self, path, metadata):
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as error:
            raise EngineUnavailable(
                "Install the ONNX engine with: pip install 'akiki[onnx]'"
            ) from error
        path = path.with_suffix(".onnx")
        verify_file(path, metadata["onnx_sha256"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise EngineUnavailable(f"ONNX Runtime could not load the model: {error}") from error
        if {v.name for v in self.session.get_inputs()} != {"audio_with_context", "h"}:
            raise ValueError("Unexpected ONNX inputs")

    def infer(self, window, state):
        return self.session.run(
            ["probability", "h_out"], {"audio_with_context": window[None], "h": state}
        )

    def close(self):
        super().close()
        self.session = None


class PytorchBackend(Backend):
    def __init__(self, path, metadata):
        try:
            import torch
        except (ImportError, OSError) as error:
            raise EngineUnavailable(
                "Install the PyTorch engine with: pip install 'akiki[pytorch]'"
            ) from error
        from ._torch import AkikiModel

        path = path.with_suffix(".pt")
        expected = metadata.get("pytorch_sha256")
        # The frozen release manifest already records the portable checkpoint hash.
        if (
            expected is None
            and path.resolve() == Path(__file__).resolve().parent / "models/akiki-vad.pt"
        ):
            release = json.loads((path.parent / "release.json").read_text())
            expected = release["files"][path.name]["sha256"]
        if not path.is_file():
            raise EngineUnavailable(f"Local PyTorch checkpoint is missing: {path}")
        if expected is None:
            raise ValueError("Custom PyTorch models require pytorch_sha256 in their JSON sidecar")
        verify_file(path, expected)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["spec"] != metadata["spec"]:
            raise ValueError("PyTorch checkpoint spec differs from its metadata")
        self.torch = torch
        self.model = AkikiModel(metadata["spec"]["hidden"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def infer(self, window, state):
        with self.torch.inference_mode():
            probability, hidden = self.model(
                self.torch.from_numpy(window[None]), self.torch.from_numpy(state)
            )
        return probability.numpy(), hidden.numpy()

    def close(self):
        super().close()
        self.model = None


def library_name(stem="akiki_vad_engine"):
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def library_candidates(library_path=None, stem="akiki_vad_engine", variable="AKIKI_VAD_LIBRARY"):
    explicit = library_path or os.environ.get(variable)
    if explicit:
        yield Path(explicit).expanduser()
        return
    name = library_name(stem)
    yield Path(__file__).parent / "native" / name
    # The offline build helper keeps artifacts inside the Python workspace.
    source_root = Path(__file__).resolve().parents[2]
    yield source_root / ".cache/native-target/release" / name
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    roots = [
        Path.home() / "projects/Akiki/akiki-inference",
        Path("/projects/Akiki/akiki-inference"),
    ]
    if os.environ.get("AKIKI_INFERENCE_DIR"):
        roots.insert(0, Path(os.environ["AKIKI_INFERENCE_DIR"]).expanduser())
    targets = ([Path(target_dir).expanduser()] if target_dir else []) + [
        r / "target" for r in roots
    ]
    for target in targets:
        if os.environ.get("CARGO_BUILD_TARGET"):
            target = target / os.environ["CARGO_BUILD_TARGET"]
        for profile in ("release", "debug"):
            yield target / profile / name


class AkikiBackend(Backend):
    def __init__(
        self,
        path,
        metadata,
        library_path=None,
        model_precision: ModelPrecision = ModelPrecision.INT8,
        cache_dir=None,
    ):
        folder = path.with_name(path.stem + "-native")
        if not (folder / "export.json").is_file():
            raise EngineUnavailable(f"Local Akiki model directory is missing: {folder}")
        receipt = json.loads((folder / "export.json").read_text())
        verify_file(folder / "config.json", receipt["config_sha256"])
        verify_file(folder / "model.safetensors", receipt["model_safetensors_sha256"])
        config = json.loads((folder / "config.json").read_text())
        if config.get("source_model_id", config.get("model_id")) != metadata.get("model_id"):
            raise ValueError("Akiki model identity differs from the requested model")
        if config.get("torch_dtype", "float32") != "float32":
            raise ValueError("Precision selection requires original FP32 source weights")
        spec = metadata["spec"]
        self.chunk_samples = spec["chunk_samples"]
        self.hidden = spec["hidden"]
        expected_shape = (self.chunk_samples, spec["context_samples"], self.hidden)
        failures = []
        for candidate in dict.fromkeys(library_candidates(library_path)):
            if not candidate.is_file():
                continue
            try:
                lib = ctypes.CDLL(str(candidate.resolve()))
                self._bind(lib)
            except (OSError, AttributeError) as error:
                failures.append(f"{candidate}: {error}")
                continue
            model_path = folder
            if model_precision is not ModelPrecision.FP32:
                from ._quantization import cached_model

                def validate_source():
                    original = lib.akiki_vad_create(str(folder.resolve()).encode("utf-8"))
                    if not original:
                        raise EngineUnavailable("Akiki engine cannot load the original FP32 model")
                    lib.akiki_vad_free(original)

                try:
                    model_path = cached_model(
                        folder, receipt, model_precision, cache_dir, validate_source
                    )
                except EngineUnavailable as error:
                    failures.append(f"{candidate}: {error}")
                    continue
            handle = lib.akiki_vad_create(str(model_path.resolve()).encode("utf-8"))
            if not handle:
                failures.append(f"{candidate}: cannot load this model format")
                continue
            actual_shape = (
                lib.akiki_vad_hop_samples(handle),
                lib.akiki_vad_context_samples(handle),
                lib.akiki_vad_hidden_size(handle),
            )
            if actual_shape != expected_shape:
                lib.akiki_vad_free(handle)
                failures.append(f"{candidate}: model dimensions differ from metadata")
                continue
            self.lib, self.handle = lib, handle
            self.library_path = candidate.resolve()
            self.model_path = model_path.resolve()
            self._finalizer = weakref.finalize(self, lib.akiki_vad_free, handle)
            return
        reason = "; ".join(failures) or "no local shared library found"
        raise EngineUnavailable(
            f"No compatible Akiki engine: {reason}. Set AKIKI_VAD_LIBRARY to a local library"
        )

    @staticmethod
    def _bind(lib):
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        bindings = {
            "create": ([ctypes.c_char_p], pointer),
            "free": ([pointer], None),
            "reset": ([pointer], None),
            "hop_samples": ([pointer], size),
            "context_samples": ([pointer], size),
            "hidden_size": ([pointer], size),
            "chunk_len": ([pointer, pointer, size], ctypes.c_float),
            "state_len": ([pointer, pointer, size], ctypes.c_int),
        }
        for name, (args, result) in bindings.items():
            fn = getattr(lib, "akiki_vad_" + name)
            fn.argtypes, fn.restype = args, result

    def reset(self):
        super().reset()
        self.lib.akiki_vad_reset(self.handle)

    def infer(self, window, state):
        chunk = np.ascontiguousarray(window[-self.chunk_samples :], dtype=np.float32)
        probability = self.lib.akiki_vad_chunk_len(self.handle, chunk.ctypes.data, len(chunk))
        hidden = np.empty_like(state)
        if self.lib.akiki_vad_state_len(self.handle, hidden.ctypes.data, self.hidden):
            raise RuntimeError("Akiki engine could not return its hidden state")
        return probability, hidden

    def close(self):
        super().close()
        self._finalizer()


def load_engine(
    path,
    metadata,
    engine: Engine,
    library_path,
    model_precision: ModelPrecision | None = None,
    cache_dir=None,
) -> tuple[Backend, Engine, dict[Engine, str]]:
    errors: dict[Engine, str] = {}
    choices = (Engine.AKIKI, Engine.ONNX, Engine.PYTORCH) if engine is Engine.AUTO else (engine,)
    if model_precision not in (None, ModelPrecision.FP32):
        choices = (Engine.AKIKI,)
    for name in choices:
        try:
            if name is Engine.AKIKI:
                backend = AkikiBackend(
                    path, metadata, library_path, model_precision or ModelPrecision.INT8, cache_dir
                )
            elif name is Engine.ONNX:
                backend = OnnxBackend(path, metadata)
            else:
                backend = PytorchBackend(path, metadata)
            return backend, name, errors
        except EngineUnavailable as error:
            errors[name] = str(error)
    reason = (
        f"model_precision={model_precision.value!r} requires a compatible Akiki engine. "
        if model_precision not in (None, ModelPrecision.FP32)
        else "No usable local VAD engine. "
    )
    raise RuntimeError(reason + "; ".join(f"{k}: {v}" for k, v in errors.items()))
