import hashlib
import json

import numpy as np
import pytest

from evaluation import embed_corpus as ec

DIM = 8


class FakeEncoder:
    """Deterministic unit vectors derived from the text, with a call log."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([vector_for(t) for t in texts])


def vector_for(text):
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def corpus(tmp_path):
    doc_ids = [1, 1, 2, 3, 3, 3, 4]
    titles = {1: "זכויות עובדים", 2: "חופשה שנתית", 3: "דמי לידה", 4: "חופשה שנתית"}
    columns = {
        "doc_id": {},
        "title": {},
        "content": {},
        "license": {},
        "link": {},
    }
    for i, (label, doc_id) in enumerate(zip([0, 1, 2, 5, 6, 7, 9], doc_ids, strict=True)):
        key = str(label)
        columns["doc_id"][key] = doc_id
        columns["title"][key] = titles[doc_id]
        columns["content"][key] = f"פסקה {i} של מסמך {doc_id}"
        columns["license"][key] = "CC"
        columns["link"][key] = f"https://example.org/{doc_id}"
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(columns, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "Fake_Model_v1"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def run(corpus, model_dir, out_dir, encoder, shard_size=3, limit=None):
    ec.embed_corpus(corpus, model_dir, out_dir, encoder, shard_size=shard_size, limit=limit)


def test_embed_and_finalize_keep_corpus_order(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())

    manifest = ec.finalize(corpus, model_dir, out)

    df = ec.load_corpus(corpus)
    paragraphs = np.load(out / "paragraphs.npy")
    titles = np.load(out / "titles.npy")
    assert paragraphs.shape == (7, DIM)
    np.testing.assert_array_equal(paragraphs, np.stack([vector_for(c) for c in df["content"]]))
    np.testing.assert_array_equal(
        titles, np.stack([vector_for(t) for t in ["זכויות עובדים", "חופשה שנתית", "דמי לידה"]])
    )
    assert manifest["paragraphs"]["count"] == 7
    assert manifest["paragraphs"]["documents"] == 4
    assert manifest["titles"]["count"] == 3
    assert manifest["dim"] == DIM
    assert manifest["paragraphs"]["order_sha256"] == ec.paragraph_order_digest(
        df["doc_id"], df["content"]
    )
    assert manifest["model"]["safetensors_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert manifest["corpus"]["sha256"] == ec.file_sha256(corpus)
    assert manifest["script_sha256"] == ec.file_sha256(ec.Path(ec.__file__))
    assert manifest["max_norm_deviation"] < 1e-6
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_resume_skips_complete_shards(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    encoder = FakeEncoder()

    run(corpus, model_dir, out, encoder)

    assert encoder.calls == []


def test_resume_recomputes_only_a_corrupt_shard(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    (out / "shards" / "paragraphs_00001.npy").write_bytes(b"truncated")
    encoder = FakeEncoder()

    run(corpus, model_dir, out, encoder)

    df = ec.load_corpus(corpus)
    assert encoder.calls == [list(df["content"][3:6])]


def test_leftover_temporary_file_is_ignored(corpus, model_dir, tmp_path):
    shards = tmp_path / "out" / "shards"
    shards.mkdir(parents=True)
    (shards / "paragraphs_00000.npy.tmp").write_bytes(b"partial")

    run(corpus, model_dir, tmp_path / "out", FakeEncoder())

    assert not list(shards.glob("*.tmp"))
    assert np.load(shards / "paragraphs_00000.npy").shape == (3, DIM)


def test_different_configuration_is_refused(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder(), shard_size=3)

    with pytest.raises(ec.EmbeddingError):
        run(corpus, model_dir, out, FakeEncoder(), shard_size=4)


def test_runs_are_recorded(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    ec.embed_corpus(corpus, model_dir, out, FakeEncoder(), shard_size=3, run_info={"device": "x"})
    ec.embed_corpus(corpus, model_dir, out, FakeEncoder(), shard_size=3, run_info={"device": "y"})

    run_file = ec.load_run_file(out / "shards")

    assert [r["device"] for r in run_file["runs"]] == ["x", "y"]


def test_limit_embeds_first_paragraphs_only(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder(), limit=4)

    manifest = ec.finalize(corpus, model_dir, out)

    assert manifest["corpus"]["limit"] == 4
    assert manifest["paragraphs"]["count"] == 4
    assert manifest["titles"]["count"] == 3
    assert np.load(out / "paragraphs.npy").shape == (4, DIM)


def test_finalize_refuses_missing_shard(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    (out / "shards" / "titles_00000.npy").unlink()

    with pytest.raises(ec.EmbeddingError):
        ec.finalize(corpus, model_dir, out)


def test_verify_passes_on_matching_embeddings(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    ec.finalize(corpus, model_dir, out)

    result = ec.verify(corpus, model_dir, out, sample_size=7, encode=FakeEncoder())

    assert result["ok"] is True
    assert result["sampled_paragraphs"] == 7
    assert result["sampled_titles"] == 3
    assert result["min_cosine"] > 0.9999
    assert result["script_matches_manifest"] is True


def test_verify_fails_on_a_perturbed_vector(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    ec.finalize(corpus, model_dir, out)
    paragraphs = np.load(out / "paragraphs.npy")
    paragraphs[2] = vector_for("something else")
    np.save(out / "paragraphs.npy", paragraphs)

    result = ec.verify(corpus, model_dir, out, sample_size=7, encode=FakeEncoder())

    assert result["ok"] is False
    assert result["checks"]["sample_cosine"] is False


def test_verify_fails_on_a_different_corpus(corpus, model_dir, tmp_path):
    out = tmp_path / "out"
    run(corpus, model_dir, out, FakeEncoder())
    ec.finalize(corpus, model_dir, out)
    corpus.write_text(corpus.read_text(encoding="utf-8").replace("פסקה 0", "פסקה X"), "utf-8")

    result = ec.verify(corpus, model_dir, out, sample_size=2, encode=FakeEncoder())

    assert result["checks"]["corpus_sha256"] is False
    assert result["checks"]["paragraph_order"] is False
    assert result["ok"] is False


def test_title_order_digest_depends_on_order():
    assert ec.title_order_digest(["א", "ב"]) != ec.title_order_digest(["ב", "א"])
