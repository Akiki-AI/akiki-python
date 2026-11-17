"""Trained affine parameters must survive restarts and failed cache writes."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from akiki import _affine_cache
from akiki.decision import AffineLayer, BaseModel, Specialist


@pytest.fixture
def base(tmp_path):
    return BaseModel(tmp_path / "models--org--encoder/snapshots/first", 32, 4)


def specialist(cache_dir, **overrides):
    return Specialist(
        **{
            "intents": {"question", "task"},
            "examples": {"Why?": "question", "Restart.": "task"},
            "cache_dir": cache_dir,
            **overrides,
        }
    )


def layer(width=4, scale=1, dtype=np.float32):
    return AffineLayer(np.eye(width, dtype=dtype) * scale, np.full(width, scale, dtype=dtype))


def test_affine_survives_a_new_python_process_without_optional_runtimes(tmp_path, base):
    cache = tmp_path / "cache"
    first = specialist(cache)
    first.affine_layers[base] = layer(dtype=np.float16)
    saved = next(cache.rglob("*.npz"))
    timestamp = saved.stat().st_mtime_ns
    code = """
import importlib.abc
import sys
from pathlib import Path

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'huggingface_hub', 'safetensors', 'akiki_training'
        }:
            raise AssertionError(f'Unexpected optional import: {fullname}')

sys.meta_path.insert(0, BlockRuntime())
import numpy as np
from akiki.decision import BaseModel, Specialist
base = BaseModel(Path(sys.argv[1]), 32, 4)
specialist = Specialist(
    intents={'task', 'question'},
    examples={'Restart.': 'task', 'Why?': 'question'},
    cache_dir=sys.argv[2],
)
affine = specialist.affine_layers[base]
assert affine.weight.dtype == np.float16
assert affine.bias.dtype == np.float16
np.testing.assert_array_equal(affine.weight, np.eye(4))
np.testing.assert_array_equal(affine.bias, np.ones(4))
assert list(specialist.affine_layers) == [base]
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(base.path), str(cache)],
        cwd=tmp_path,
        env=os.environ
        | {"HF_HUB_OFFLINE": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert saved.stat().st_mtime_ns == timestamp


def test_missing_lookup_does_not_create_directories_or_a_layer(tmp_path, base):
    cache = tmp_path / "missing-cache"
    item = specialist(cache)
    assert item.affine_layers.get(base) is None
    assert len(item.affine_layers) == 0
    assert not cache.exists()
    with pytest.raises(KeyError):
        item.affine_layers[base]


@pytest.mark.parametrize("change", ["intents", "example_text", "example_label"])
def test_changed_specialist_data_isolates_layers_and_preserves_old_training(tmp_path, base, change):
    cache = tmp_path / "cache"
    item = specialist(cache)
    item.affine_layers[base] = layer()
    saved = next(cache.rglob("*.npz"))
    original = saved.read_bytes()
    if change == "intents":
        item.intents.add("greeting")
    elif change == "example_text":
        item.examples["When?"] = "question"
    else:
        item.examples["Why?"] = "task"
    assert item.affine_layers.get(base) is None
    assert len(item.affine_layers) == 0
    assert saved.read_bytes() == original
    item.affine_layers[base] = layer(scale=2)
    np.testing.assert_array_equal(item.affine_layers[base].bias, np.full(4, 2))
    original_specialist = specialist(cache)
    np.testing.assert_array_equal(original_specialist.affine_layers[base].bias, np.ones(4))
    assert len(list(cache.rglob("*.npz"))) == 2
    item.intents = original_specialist.intents.copy()
    item.examples = original_specialist.examples.copy()
    np.testing.assert_array_equal(item.affine_layers[base].bias, np.ones(4))


@pytest.mark.parametrize("change", ["revision", "vocab_size", "embedding_dim"])
def test_base_model_identity_separates_entries(tmp_path, base, change):
    other = {
        "revision": BaseModel(base.path.with_name("second"), 32, 4),
        "vocab_size": BaseModel(base.path, 40, 4),
        "embedding_dim": BaseModel(base.path, 32, 8),
    }[change]
    item = specialist(tmp_path / "cache")
    item.affine_layers[base] = layer()
    assert item.affine_layers.get(other) is None
    item.affine_layers[other] = layer(other.embedding_dim, scale=2)
    assert set(item.affine_layers) == {base, other}
    np.testing.assert_array_equal(item.affine_layers[base].bias, np.ones(4))
    np.testing.assert_array_equal(item.affine_layers[other].bias, np.full(other.embedding_dim, 2))


def test_other_specialist_data_has_a_separate_cache(tmp_path, base):
    cache = tmp_path / "cache"
    first = specialist(cache)
    first.affine_layers[base] = layer()
    second = specialist(cache, intents={"greeting"}, examples={"Hello.": "greeting"})
    assert second.affine_layers.get(base) is None


def test_default_location_honors_akiki_cache_and_explicit_override(tmp_path, monkeypatch, base):
    default = tmp_path / "default"
    monkeypatch.setenv("AKIKI_CACHE_DIR", str(default))
    first = specialist(None)
    assert not default.exists()
    first.affine_layers[base] = layer()
    assert len(list((default / "affines").rglob("*.npz"))) == 1
    explicit = tmp_path / "explicit"
    second = specialist(explicit)
    assert second.affine_layers.get(base) is None
    second.affine_layers[base] = layer(scale=2)
    assert len(list(explicit.rglob("*.npz"))) == 1
    np.testing.assert_array_equal(first.affine_layers[base].bias, np.ones(4))


def test_initial_layers_are_persisted_and_loaded_edits_require_reassignment(tmp_path, base):
    cache = tmp_path / "cache"
    first = specialist(cache, affine_layers={base: layer()})
    loaded = first.affine_layers[base]
    loaded.bias[:] = 3
    second = specialist(cache)
    np.testing.assert_array_equal(second.affine_layers[base].bias, np.ones(4))
    first.affine_layers[base] = loaded
    np.testing.assert_array_equal(second.affine_layers[base].bias, np.full(4, 3))


@pytest.mark.parametrize("stage", ["write", "replace"])
def test_failed_save_preserves_previous_expensive_layer_and_cleans_temporary_files(
    tmp_path, monkeypatch, base, stage
):
    cache = tmp_path / "cache"
    item = specialist(cache)
    item.affine_layers[base] = layer()
    saved = next(cache.rglob("*.npz"))
    original = saved.read_bytes()

    def fail_write(stream, **arrays):
        stream.write(b"partial")
        raise OSError("Simulated write failure")

    def fail_replace(source, destination):
        raise OSError("Simulated replace failure")

    if stage == "write":
        monkeypatch.setattr(_affine_cache.np, "savez", fail_write)
    else:
        monkeypatch.setattr(_affine_cache.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="Cannot save trained affine.*"):
        item.affine_layers[base] = layer(scale=2)
    assert saved.read_bytes() == original
    np.testing.assert_array_equal(item.affine_layers[base].bias, np.ones(4))
    assert not list(cache.rglob(".tmp-*"))


@pytest.mark.parametrize("damage", ["archive", "metadata", "missing_array", "shape", "object"])
def test_damaged_entries_raise_and_remain_on_disk_instead_of_becoming_cache_misses(
    tmp_path, base, damage
):
    cache = tmp_path / "cache"
    item = specialist(cache)
    item.affine_layers[base] = layer()
    saved = next(cache.rglob("*.npz"))
    if damage == "archive":
        saved.write_bytes(b"damaged archive")
    else:
        with np.load(saved, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if damage == "metadata":
            metadata = json.loads(arrays["metadata"].item())
            metadata["version"] += 1
            arrays["metadata"] = np.array(json.dumps(metadata))
        elif damage == "missing_array":
            del arrays["bias"]
        elif damage == "shape":
            arrays["weight"] = np.eye(8, dtype=np.float32)
        else:
            arrays["weight"] = np.array([[object()]], dtype=object)
        np.savez(saved, **arrays)
    damaged = saved.read_bytes()
    with pytest.raises(ValueError, match="Cannot read trained affine cache entry"):
        specialist(cache).affine_layers.get(base)
    assert saved.read_bytes() == damaged
    assert list(cache.rglob("*.npz")) == [saved]


@pytest.mark.parametrize(
    "invalid,error",
    [
        (object(), TypeError),
        (AffineLayer(np.eye(3), np.ones(4)), ValueError),
        (AffineLayer(np.eye(4), np.ones(3)), ValueError),
        (AffineLayer(np.eye(4, dtype=int), np.ones(4)), ValueError),
        (AffineLayer(np.eye(4), np.full(4, np.nan)), ValueError),
    ],
)
def test_invalid_layers_do_not_create_cache_entries(tmp_path, base, invalid, error):
    cache = tmp_path / "cache"
    with pytest.raises(error):
        specialist(cache).affine_layers[base] = invalid
    assert not cache.exists()


def test_concurrent_saves_leave_one_complete_valid_layer(tmp_path, base):
    cache = tmp_path / "cache"

    def save(scale):
        specialist(cache).affine_layers[base] = layer(scale=scale)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(1, 9)))
    final = specialist(cache).affine_layers[base]
    scale = final.bias[0]
    assert scale in range(1, 9)
    np.testing.assert_array_equal(final.weight, np.eye(4) * scale)
    np.testing.assert_array_equal(final.bias, np.full(4, scale))
    assert len(list(cache.rglob("*.npz"))) == 1
    assert not list(cache.rglob(".tmp-*"))


def test_explicit_deletion_removes_only_the_selected_entry(tmp_path, base):
    cache = tmp_path / "cache"
    item = specialist(cache)
    other = BaseModel(base.path.with_name("second"), 32, 4)
    item.affine_layers[base] = layer()
    item.affine_layers[other] = layer(scale=2)
    del item.affine_layers[base]
    restored = specialist(cache)
    assert restored.affine_layers.get(base) is None
    assert list(restored.affine_layers) == [other]
    with pytest.raises(KeyError):
        del restored.affine_layers[base]
