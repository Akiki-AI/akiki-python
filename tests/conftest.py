import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from akiki import VAD, Engine


@pytest.fixture
def require_rust():
    try:
        VAD(engine=Engine.RUST).close()
    except RuntimeError:
        if os.environ.get("AKIKI_VAD_TEST_RUST") == "1":
            raise
        pytest.skip("No compatible local Rust library")


@pytest.fixture(scope="session")
def require_minilm_library():
    from akiki.minilm import _load_library

    try:
        return _load_library()[1]
    except RuntimeError:
        if os.environ.get("AKIKI_MINILM_TEST_RUST") == "1":
            raise
        pytest.skip("No compatible local MiniLM library")


@pytest.fixture(scope="session")
def minilm_model():
    # Model weights are not bundled; real-model tests use an existing local copy.
    value = os.environ.get("AKIKI_MINILM_MODEL")
    if not value:
        pytest.skip("Set AKIKI_MINILM_MODEL to a local MiniLM checkpoint or Ariadne variant folder")
    return Path(value)


TINY = SimpleNamespace(hidden=64, width=128, layers=2, vocab=40, positions=48, classes=5)


def write_safetensors(path, tensors):
    """Write FP32 tensors in the safetensors format."""
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        data = np.ascontiguousarray(tensors[name], dtype="<f4")
        header[name] = {
            "dtype": "F32",
            "shape": list(data.shape),
            "data_offsets": [offset, offset + data.nbytes],
        }
        blobs.append(data.tobytes())
        offset += data.nbytes
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))


def tiny_tokenizer():
    """Word-level vocabulary ``w4``..; ``[CLS]`` is 1 and ``[SEP]`` 2. Truncation
    and padding are enabled, as in published MiniLM tokenizers."""
    special = [
        {
            "id": i,
            "content": token,
            "special": True,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }
        for i, token in enumerate(["[PAD]", "[CLS]", "[SEP]", "[UNK]"])
    ]
    vocab = {s["content"]: s["id"] for s in special} | {f"w{i}": i for i in range(4, TINY.vocab)}

    def token(name):
        return {"SpecialToken": {"id": name, "type_id": 0}}

    def sequence(name):
        return {"Sequence": {"id": name, "type_id": 0}}

    return {
        "version": "1.0",
        "truncation": {"max_length": 4, "strategy": "LongestFirst", "stride": 0},
        "padding": {
            "strategy": {"Fixed": 8},
            "direction": "Right",
            "pad_to_multiple_of": None,
            "pad_id": 0,
            "pad_type_id": 0,
            "pad_token": "[PAD]",
        },
        "added_tokens": special,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [token("[CLS]"), sequence("A"), token("[SEP]")],
            "pair": [token("[CLS]"), sequence("A"), token("[SEP]"), sequence("B"), token("[SEP]")],
            "special_tokens": {
                name: {"id": name, "ids": [vocab[name]], "tokens": [name]}
                for name in ["[CLS]", "[SEP]"]
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }


def tiny_checkpoint(folder, rng):
    """Write a small random BERT checkpoint and tokenizer to ``folder``."""
    d = TINY.hidden

    def uniform(*shape, scale, offset=0.0):
        return offset + scale * rng.uniform(-1, 1, shape)

    tensors = {
        "embeddings.word_embeddings.weight": uniform(TINY.vocab, d, scale=0.5),
        "embeddings.position_embeddings.weight": uniform(TINY.positions, d, scale=0.1),
        "embeddings.token_type_embeddings.weight": uniform(2, d, scale=0.1),
    }
    norms = ["embeddings.LayerNorm"]
    for i in range(TINY.layers):
        p = f"encoder.layer.{i}"
        for name, rows, cols in [
            ("attention.self.query", d, d),
            ("attention.self.key", d, d),
            ("attention.self.value", d, d),
            ("attention.output.dense", d, d),
            ("intermediate.dense", TINY.width, d),
            ("output.dense", d, TINY.width),
        ]:
            tensors[f"{p}.{name}.weight"] = uniform(rows, cols, scale=0.2)
            tensors[f"{p}.{name}.bias"] = uniform(rows, scale=0.05)
        norms += [f"{p}.attention.output.LayerNorm", f"{p}.output.LayerNorm"]
    for p in norms:
        tensors[f"{p}.weight"] = uniform(d, scale=0.1, offset=1.0)
        tensors[f"{p}.bias"] = uniform(d, scale=0.1)
    folder.mkdir(parents=True)
    write_safetensors(folder / "model.safetensors", tensors)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "hidden_size": d,
                "intermediate_size": TINY.width,
                "num_attention_heads": 4,
                "num_hidden_layers": TINY.layers,
                "max_position_embeddings": TINY.positions,
                "vocab_size": TINY.vocab,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
                "hidden_act": "gelu",
                "position_embedding_type": "absolute",
            }
        )
    )
    (folder / "tokenizer.json").write_text(json.dumps(tiny_tokenizer()))


@pytest.fixture(scope="session")
def tiny_minilm(tmp_path_factory):
    """Tiny random MiniLM checkpoints and Ariadne-style variants; no real assets."""
    root = tmp_path_factory.mktemp("tiny-minilm")
    rng = np.random.default_rng(1729)
    tiny_checkpoint(root / "plain", rng)
    d = TINY.hidden
    head = {
        "head.weight": rng.uniform(-1, 1, (TINY.classes, d)),
        "head.bias": rng.uniform(-0.1, 0.1, TINY.classes),
    }
    labels = tuple(f"label-{i}" for i in range(TINY.classes))
    # Classifier layout: base_model/, an identity input affine with a head, and labels.
    variant = root / "variant"
    shutil.copytree(root / "plain", variant / "base_model")
    write_safetensors(
        variant / "adapter.safetensors",
        {"interface.proj.weight": np.eye(d), "interface.proj.bias": np.zeros(d), **head},
    )
    (variant / "model_config.json").write_text(json.dumps({"labels": labels}))
    # Retrieval layout: a trained affine stored as proj.*, no head, chunked at max_length.
    retrieval = root / "retrieval"
    shutil.copytree(root / "plain", retrieval / "base_model")
    write_safetensors(
        retrieval / "adapter.safetensors",
        {
            "proj.weight": np.eye(d) + rng.uniform(-0.05, 0.05, (d, d)),
            "proj.bias": rng.uniform(-0.05, 0.05, d),
        },
    )
    (retrieval / "model_config.json").write_text(json.dumps({"max_length": 8}))
    return SimpleNamespace(
        plain=root / "plain",
        variant=variant,
        retrieval=retrieval,
        head=head,
        labels=labels,
        hidden=d,
        positions=TINY.positions,
        vocab=TINY.vocab,
    )
