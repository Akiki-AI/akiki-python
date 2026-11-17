import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from akiki import vad
from akiki.decision import MiniLM, ModelPrecision


def words(n, start=0):
    return " ".join(f"w{4 + (start + i) % 36}" for i in range(n))


def fake_model(folder):
    (folder / "base_model").mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (folder / "base_model" / name).write_text("{}")
    return folder


def test_missing_files_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        MiniLM(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        MiniLM(fake_model(tmp_path), tmp_path / "missing.safetensors")


@pytest.mark.parametrize(
    "options",
    [
        dict(threads=0),
        dict(threads=257),
        dict(threads=1.5),
        dict(threads=True),
    ],
)
def test_invalid_options(tmp_path, options):
    with pytest.raises(ValueError):
        MiniLM(fake_model(tmp_path), **options)


@pytest.mark.parametrize(
    "precision",
    [
        *(precision.value for precision in ModelPrecision),
        "fp8",
        *vad.ModelPrecision,
        vad.FFTPrecision.FP32,
        vad.Engine.AKIKI,
        None,
        1,
        [],
    ],
)
def test_model_precision_requires_the_decision_enum(tmp_path, precision):
    with pytest.raises(TypeError, match=r"akiki\.decision\.ModelPrecision"):
        MiniLM(tmp_path, model_precision=precision)


def test_missing_library_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AKIKI_MINILM_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="AKIKI_MINILM_LIBRARY"):
        MiniLM(fake_model(tmp_path))


def test_engine_reports_why_a_model_cannot_load(tmp_path, require_minilm_library):
    with pytest.raises(ValueError, match="MiniLM engine cannot load.*model_type"):
        MiniLM(fake_model(tmp_path), library_path=require_minilm_library)


@pytest.fixture(scope="module")
def plain(tiny_minilm, require_minilm_library):
    with MiniLM(tiny_minilm.plain, model_precision=ModelPrecision.FP32) as model:
        yield model


@pytest.fixture(scope="module")
def variant(tiny_minilm, require_minilm_library):
    with MiniLM(tiny_minilm.variant, model_precision=ModelPrecision.FP32) as model:
        yield model


def test_plain_checkpoint_embeds_without_a_head(plain, tiny_minilm):
    assert (plain.adapter_path, plain.labels, plain.chunk_tokens) == (None, None, None)
    assert (plain.embedding_size, plain.head_size, plain.max_tokens) == (
        tiny_minilm.hidden,
        0,
        tiny_minilm.positions,
    )
    vector = plain.embed(words(9))
    assert vector.dtype == np.float32 and vector.shape == (tiny_minilm.hidden,)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert plain.encode(words(9)).logits is None
    texts = [words(3), words(9, 5), ""]
    batch = plain.embed(texts)
    assert batch.shape == (3, tiny_minilm.hidden)
    np.testing.assert_array_equal(batch[1], plain.embed(texts[1]))
    assert plain.embed([]).shape == (0, tiny_minilm.hidden)
    with pytest.raises(ValueError, match="no classification head"):
        plain.classify(words(3))


def test_variant_folder_loads_its_adapter_and_labels(variant, plain, tiny_minilm):
    assert variant.model_path == tiny_minilm.variant / "base_model"
    assert variant.adapter_path == tiny_minilm.variant / "adapter.safetensors"
    assert (variant.head_size, variant.labels) == (len(tiny_minilm.labels), tiny_minilm.labels)
    text = words(17, 3)
    # The adapter's identity affine leaves the encoder exactly as it was.
    np.testing.assert_array_equal(variant.embed(text), plain.embed(text))
    encoding = variant.encode(text)
    w, b = tiny_minilm.head["head.weight"], tiny_minilm.head["head.bias"]
    np.testing.assert_allclose(encoding.logits, w @ encoding.embedding + b, atol=1e-5)
    result = variant.classify(text)
    assert result.index == int(np.argmax(encoding.logits))
    assert result.label == tiny_minilm.labels[result.index]
    expected = np.exp(encoding.logits - encoding.logits.max())
    np.testing.assert_allclose(result.probabilities, expected / expected.sum(), rtol=1e-6)
    np.testing.assert_allclose(result.logits, encoding.logits)


def test_adapter_selection(tiny_minilm, require_minilm_library):
    with MiniLM(tiny_minilm.variant / "base_model") as model:
        assert (model.adapter_path, model.head_size) == (None, 0)
    with MiniLM(tiny_minilm.plain, tiny_minilm.variant / "adapter.safetensors") as model:
        assert (model.head_size, model.labels) == (len(tiny_minilm.labels), None)
        assert model.classify(words(4)).label is None


def test_tokenizer_never_truncates_or_pads(plain):
    assert plain.tokenize(words(9)) == [1, *range(4, 13), 2]
    assert plain.tokenize(words(9), special_tokens=False) == list(range(4, 13))
    assert plain.tokenize(words(2)) == [1, 4, 5, 2]
    assert len(plain.tokenize(words(3000), special_tokens=False)) == 3000


def test_long_text_is_rejected_unless_chunked(plain, tiny_minilm):
    plain.embed(words(tiny_minilm.positions - 2))
    with pytest.raises(ValueError, match="chunk_tokens"):
        plain.embed(words(tiny_minilm.positions - 1))


def test_chunking_combines_weighted_chunk_embeddings(plain, tiny_minilm, require_minilm_library):
    with MiniLM(tiny_minilm.plain, model_precision=ModelPrecision.FP32, chunk_tokens=8) as chunked:
        text = words(20, 7)
        tokens = plain.tokenize(text, special_tokens=False)
        pieces = [tokens[i : i + 6] for i in range(0, len(tokens), 6)]
        chunks = [[1, *piece, 2] for piece in pieces]
        weights = [len(piece) for piece in pieces]
        assert weights == [6, 6, 6, 2]
        vectors = np.stack([plain.encode_chunks([chunk], [1]).embedding for chunk in chunks])
        expected = (vectors * np.array(weights)[:, None]).sum(0)
        np.testing.assert_allclose(
            chunked.embed(text), expected / np.linalg.norm(expected), atol=1e-6
        )
        np.testing.assert_array_equal(
            chunked.embed(text), plain.encode_chunks(chunks, weights).embedding
        )
        long = words(3000)
        assert np.isfinite(chunked.embed(long)).all()
        np.testing.assert_allclose(chunked.embed(""), plain.embed(""), atol=1e-6)


def test_retrieval_variant_chunks_at_its_max_length(plain, tiny_minilm, require_minilm_library):
    with MiniLM(tiny_minilm.retrieval, model_precision=ModelPrecision.FP32) as model:
        assert (model.chunk_tokens, model.head_size) == (8, 0)
        vector = model.embed(words(5))
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
        assert np.abs(vector - plain.embed(words(5))).max() > 1e-3, "the affine had no effect"


@pytest.mark.parametrize(
    "chunks,weights",
    [
        ([], []),
        ([[1, 4, 2]], []),
        ([[1, 4, 2]], [0]),
        ([[1, 4, 2]], [float("nan")]),
        ([[]], [1]),
        ([[1] * 49], [1]),
        ([[1, 40, 2]], [1]),
        ([[1, -4, 2]], [1]),
        ([[1.0, 4.0, 2.0]], [1]),
    ],
)
def test_invalid_chunks_are_rejected(plain, chunks, weights):
    with pytest.raises(ValueError):
        plain.encode_chunks(chunks, weights)


def test_invalid_variant_settings(tiny_minilm, tmp_path, require_minilm_library):
    for chunk_tokens in (2, tiny_minilm.positions + 1, 8.0):
        with pytest.raises(ValueError, match="chunk_tokens"):
            MiniLM(tiny_minilm.plain, chunk_tokens=chunk_tokens)
    folder = tmp_path / "mislabelled"
    shutil.copytree(tiny_minilm.variant, folder)
    (folder / "model_config.json").write_text(json.dumps({"labels": ["a", "b"]}))
    with pytest.raises(ValueError, match="2 labels for 5 head outputs"):
        MiniLM(folder)


def test_every_precision_loads(variant, tiny_minilm):
    for precision in ModelPrecision:
        with MiniLM(tiny_minilm.variant, model_precision=precision, threads=2) as model:
            assert model.model_precision is precision and model.threads == 2
            if precision is not ModelPrecision.FP32:
                assert model.model_bytes < variant.model_bytes
            vector = model.embed(words(11))
            assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
            assert float(vector @ variant.embed(words(11))) > 0.98
    with MiniLM(tiny_minilm.variant) as model:
        assert model.model_precision is ModelPrecision.INT8


def test_shared_instance_is_thread_safe(variant):
    texts = [words(n, n) for n in range(1, 33)]
    expected = [variant.classify(text) for text in texts]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(variant.classify, texts)) == expected


def test_closed_model_is_unusable(tiny_minilm, require_minilm_library):
    model = MiniLM(tiny_minilm.plain, threads=1)
    model.close()
    model.close()
    for call in (
        lambda: model.embed("w4"),
        lambda: model.tokenize("w4"),
        lambda: model.encode_chunks([[1, 2]], [1]),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    with pytest.raises(RuntimeError, match="closed"):
        with model:
            pass


CLOVER_LABELS = ("chat", "question", "build", "task", "fix", "unclear")


def clover_text(message, context):
    prior = "\n".join(f"{turn['role']}: {turn['text']}" for turn in context)
    return (
        ("Conversation context:\n" + prior + "\n\n" if prior else "")
        + "Current user message: "
        + message
    )


def test_clover_matches_recorded_reference_probabilities(minilm_model):
    rows, references = (
        minilm_model / "data/test.jsonl",
        minilm_model / "results/test-predictions.jsonl",
    )
    with MiniLM(minilm_model, model_precision=ModelPrecision.FP32) as model:
        if model.labels != CLOVER_LABELS or not (rows.is_file() and references.is_file()):
            pytest.skip("AKIKI_MINILM_MODEL is not a Clover release with recorded test predictions")
        expected = {r["id"]: r for r in map(json.loads, references.read_text().splitlines())}
        for row in map(json.loads, rows.read_text().splitlines()):
            result = model.classify(clover_text(row["message"], row["context"]))
            assert result.label == expected[row["id"]]["predicted"], row["id"]
            reference = [expected[row["id"]]["probabilities"][label] for label in CLOVER_LABELS]
            np.testing.assert_allclose(
                result.probabilities, reference, atol=1e-4, err_msg=row["id"]
            )


def test_matches_pytorch_references(minilm_model):
    """Rows hold ``text`` or ``chunks`` with ``weights``, and reference embeddings
    and/or logits; see the engine's local-model tests for the format."""
    path = os.environ.get("AKIKI_MINILM_REFERENCES")
    if not path:
        pytest.skip("Set AKIKI_MINILM_REFERENCES to local PyTorch reference JSONL for this model")
    adapter = os.environ.get("AKIKI_MINILM_ADAPTER")
    with MiniLM(minilm_model, adapter, model_precision=ModelPrecision.FP32) as model:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        for row in rows:
            if "text" in row:
                if "input_ids" in row and model.chunk_tokens is None:
                    assert model.tokenize(row["text"]) == row["input_ids"], row["id"]
                encoding = model.encode(row["text"])
            else:
                encoding = model.encode_chunks(row["chunks"], row["weights"])
            if "reference_embedding" in row:
                np.testing.assert_allclose(
                    encoding.embedding, row["reference_embedding"], atol=1e-5, err_msg=row["id"]
                )
            if "reference_logits" in row:
                np.testing.assert_allclose(
                    encoding.logits, row["reference_logits"], atol=1e-4, err_msg=row["id"]
                )
        assert rows, "Empty references do not validate parity"
