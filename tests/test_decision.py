"""Exercise local encoder discovery and decision model selection without runtimes."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from akiki import _base_models
from akiki.decision import BaseModel, DecisionModel, list_base_models


def write_safetensors(path, shapes):
    """Write synthetic tensors so tests can inspect real headers without a runtime."""
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = math.prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    mounted = tmp_path / "mounted" / "hub"
    monkeypatch.setattr(_base_models, "_LOCAL_HUB_CACHE", mounted)
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "default-home"))
    return mounted


def cached_encoder(cache, repository="org/encoder", revision="first", **config):
    path = cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "vocab_size": 32,
                "hidden_size": 16,
                **config,
            }
        )
    )
    (path / "tokenizer.json").write_text("{}")
    write_safetensors(path / "model.safetensors", {})
    return path


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("bert", ["BertModel"]),
        ("modernbert", ["ModernBertForMaskedLM"]),
        ("xlm-roberta", ["XLMRobertaForSequenceClassification"]),
        ("embedding_gemma2", ["EmbeddingGemma2Model"]),
    ],
)
def test_discovery_lists_encoders_with_different_heads(tmp_path, model_type, architectures):
    path = cached_encoder(tmp_path, model_type=model_type, architectures=architectures)
    assert [model.path for model in list_base_models(tmp_path)] == [path]


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]},
        {"is_decoder": True},
        {"is_encoder_decoder": True},
        {"architectures": ["BertLMHeadModel"]},
        {"architectures": ["BertForCausalLM"]},
        {"architectures": "BertModel"},
        {"model_type": "unknown"},
    ],
)
def test_discovery_excludes_generative_and_unrecognized_models(tmp_path, config):
    cached_encoder(tmp_path, **config)
    assert list_base_models(tmp_path) == []


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "model.safetensors"])
def test_discovery_excludes_incomplete_snapshots(tmp_path, missing):
    path = cached_encoder(tmp_path)
    (path / missing).unlink()
    assert list_base_models(tmp_path) == []


@pytest.mark.parametrize("content", ["{", "[]", "null"])
def test_discovery_skips_malformed_metadata(tmp_path, content):
    path = cached_encoder(tmp_path)
    (path / "config.json").write_text(content)
    assert list_base_models(tmp_path) == []


@pytest.mark.parametrize("weight_format", ["safetensors", "bin"])
def test_discovery_requires_every_indexed_weight_shard(tmp_path, weight_format):
    path = cached_encoder(tmp_path)
    (path / "model.safetensors").unlink()
    stem = "model" if weight_format == "safetensors" else "pytorch_model"
    shards = [f"{stem}-{i:05}-of-00002.{weight_format}" for i in (1, 2)]
    (path / f"{stem}.{weight_format}.index.json").write_text(
        json.dumps({"weight_map": dict(zip(("first", "second"), shards))})
    )
    for name in shards:
        write_safetensors(path / name, {})
    assert [model.path for model in list_base_models(tmp_path)] == [path]
    (path / shards[1]).unlink()
    assert list_base_models(tmp_path) == []


def test_discovery_ignores_broken_cache_symlinks(tmp_path):
    path = cached_encoder(tmp_path)
    weights = path / "model.safetensors"
    weights.unlink()
    weights.symlink_to(tmp_path / "missing-blob")
    assert list_base_models(tmp_path) == []


def test_descriptors_expose_input_embedding_dimensions_and_a_filesystem_path(tmp_path):
    path = cached_encoder(tmp_path)
    base = list_base_models(tmp_path)[0]
    assert isinstance(base, BaseModel)
    assert (base.name, base.path) == ("org/encoder", path)
    assert (base.vocab_size, base.embedding_dim, base.embedding_shape) == (32, 16, (32, 16))
    assert Path(base) == path
    assert DecisionModel(base_model=base).base_model == base


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            {"model_type": "albert", "vocab_size": 32, "hidden_size": 64, "embedding_size": 8},
            (32, 8),
        ),
        ({"model_type": "electra", "hidden_size": 64, "embedding_size": 12}, (32, 12)),
        ({"model_type": "distilbert", "hidden_size": None, "dim": 24}, (32, 24)),
        (
            {
                "model_type": "embedding_gemma2",
                "text_config": {"vocab_size": 48, "hidden_size": 12, "embedding_dim": 20},
            },
            (48, 12),
        ),
    ],
)
def test_configuration_fallback_uses_input_width_including_factorized_and_nested_models(
    tmp_path, config, expected
):
    path = cached_encoder(tmp_path, **config)
    (path / "model.safetensors").rename(path / "pytorch_model.bin")
    assert list_base_models(tmp_path)[0].embedding_shape == expected


@pytest.mark.parametrize(
    "tensor_name",
    [
        "embeddings.word_embeddings.weight",
        "model.embeddings.tok_embeddings.weight",
        "language_model.embed_tokens.weight",
        "an_unfamiliar_token_lookup.weight",
    ],
)
def test_actual_tensor_dimensions_take_precedence_over_config_width(tmp_path, tensor_name):
    path = cached_encoder(tmp_path)
    write_safetensors(path / "model.safetensors", {tensor_name: (32, 7)})
    assert list_base_models(tmp_path)[0].embedding_shape == (32, 7)


def test_named_input_table_dimensions_do_not_require_config_dimensions(tmp_path):
    path = cached_encoder(tmp_path, vocab_size=None, hidden_size=None)
    write_safetensors(path / "model.safetensors", {"word_embeddings.weight": (40, 9)})
    assert list_base_models(tmp_path)[0].embedding_shape == (40, 9)


def test_dimensions_are_read_again_at_runtime_when_a_descriptor_is_selected(tmp_path):
    path = cached_encoder(tmp_path)
    initial = list_base_models(tmp_path)[0]
    write_safetensors(path / "model.safetensors", {"word_embeddings.weight": (40, 5)})
    assert list_base_models(tmp_path)[0].embedding_shape == (40, 5)
    assert DecisionModel(base_model=initial).base_model.embedding_shape == (40, 5)


def test_unknown_llm_families_can_be_selected_by_path_or_cached_name(isolated_caches):
    path = cached_encoder(
        isolated_caches,
        repository="org/future-model",
        model_type="a_future_architecture",
        architectures=["FutureModelForCausalLM"],
    )
    write_safetensors(path / "model.safetensors", {"novel_token_table.weight": (32, 11)})
    assert list_base_models() == []  # Discovery currently lists encoders.
    assert DecisionModel(base_model=path).base_model.embedding_shape == (32, 11)
    assert DecisionModel(base_model="org/future-model").base_model.embedding_shape == (32, 11)


def test_sharded_dimensions_read_only_the_input_embedding_shard_header(tmp_path):
    path = cached_encoder(tmp_path)
    (path / "model.safetensors").unlink()
    write_safetensors(path / "tokens.safetensors", {"model.embed_tokens.weight": (32, 7)})
    (path / "other.safetensors").write_bytes(b"This shard must not be read for dimensions.")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "tokens.safetensors",
                    "other.weight": "other.safetensors",
                }
            }
        )
    )
    assert list_base_models(tmp_path)[0].embedding_shape == (32, 7)


@pytest.mark.parametrize(
    "config",
    [{"vocab_size": None}, {"vocab_size": True}, {"hidden_size": 0}, {"hidden_size": "16"}],
)
def test_unavailable_or_invalid_dimensions_are_rejected(tmp_path, config):
    path = cached_encoder(tmp_path, **config)
    assert list_base_models(tmp_path) == []
    with pytest.raises(ValueError, match="Invalid input embedding table shape"):
        DecisionModel(base_model=path)


def test_ambiguous_tensor_dimensions_are_rejected_instead_of_guessing(tmp_path):
    path = cached_encoder(tmp_path)
    write_safetensors(
        path / "model.safetensors",
        {"first_vocabulary_table": (32, 7), "second_vocabulary_table": (32, 9)},
    )
    assert list_base_models(tmp_path) == []
    with pytest.raises(ValueError, match="Multiple input embedding table dimensions"):
        DecisionModel(base_model=path)


def test_invalid_safetensors_headers_are_not_used_for_dimensions(tmp_path):
    path = cached_encoder(tmp_path)
    (path / "model.safetensors").write_bytes((10_000_000).to_bytes(8, "little"))
    assert list_base_models(tmp_path) == []


@pytest.mark.parametrize("weight_map", [None, {}, [], {"word_embeddings.weight": 123}])
def test_malformed_safetensors_index_is_reported_even_when_other_weights_exist(
    tmp_path, weight_map
):
    path = cached_encoder(tmp_path)
    (path / "model.safetensors").rename(path / "pytorch_model.bin")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    assert list_base_models(tmp_path) == []
    with pytest.raises(ValueError, match="Invalid safetensors weight index"):
        DecisionModel(base_model=path)


@pytest.mark.parametrize(
    "variable", [None, "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"]
)
def test_default_discovery_honors_cache_locations(tmp_path, monkeypatch, isolated_caches, variable):
    cache = isolated_caches
    if variable is not None:
        configured = tmp_path / "configured"
        monkeypatch.setenv(variable, str(configured))
        cache = configured
        if variable == "HF_HOME":
            cache /= "hub"
        elif variable == "XDG_CACHE_HOME":
            monkeypatch.delenv("HF_HOME")
            cache /= "huggingface/hub"
    path = cached_encoder(cache)
    assert [model.path for model in list_base_models()] == [path]
    assert list_base_models(tmp_path / "missing-cache") == []


def test_duplicate_cache_roots_are_deduplicated_and_revisions_remain_explicit(
    monkeypatch, isolated_caches
):
    first = cached_encoder(isolated_caches)
    second = cached_encoder(isolated_caches, revision="second")
    monkeypatch.setenv("HF_HUB_CACHE", str(isolated_caches))
    monkeypatch.setenv("HF_HOME", str(isolated_caches.parent))
    assert [model.path for model in list_base_models()] == [first, second]
    with pytest.raises(ValueError, match="Multiple local revisions.*Pass the desired path"):
        DecisionModel(base_model="org/encoder")
    assert DecisionModel(base_model=second).base_model.path == second


def test_base_model_can_be_selected_replaced_and_cleared(isolated_caches):
    first = cached_encoder(isolated_caches)
    second = cached_encoder(isolated_caches, repository="org/other-encoder")
    model = DecisionModel()
    assert model.base_model is None
    model.base_model = list_base_models()[0]
    assert model.base_model.path == first
    model.base_model = str(second)
    assert model.base_model.path == second
    model.base_model = None
    assert model.base_model is None
    assert DecisionModel(base_model="org/encoder").base_model.path == first


def test_invalid_base_model_does_not_replace_an_existing_selection(tmp_path, isolated_caches):
    path = cached_encoder(isolated_caches)
    model = DecisionModel(base_model=path)
    with pytest.raises(FileNotFoundError, match="Models are never downloaded"):
        model.base_model = "org/missing"
    with pytest.raises(FileNotFoundError):
        model.base_model = tmp_path / "missing"
    with pytest.raises(TypeError, match="base_model must be"):
        model.base_model = 123
    assert model.base_model.path == path


def test_discovery_and_selection_need_no_optional_runtimes_or_network(tmp_path):
    cached_encoder(tmp_path)
    code = """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'huggingface_hub', 'akiki_training', 'safetensors'
        }:
            raise AssertionError(f'Unexpected optional import: {fullname}')

def block_network(event, args):
    if event == 'socket.connect':
        raise AssertionError('Unexpected network access')

sys.meta_path.insert(0, BlockRuntime())
sys.addaudithook(block_network)
from akiki.decision import BaseModel, DecisionModel, list_base_models
paths = list_base_models(sys.argv[1])
assert len(paths) == 1
assert DecisionModel(base_model=paths[0]).base_model == paths[0]
namespace = {}
exec('from akiki.decision import *', namespace)
assert namespace['DecisionModel'] is DecisionModel
assert namespace['list_base_models'] is list_base_models
assert namespace['BaseModel'] is BaseModel
assert paths[0].embedding_shape == (32, 16)
"""
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
