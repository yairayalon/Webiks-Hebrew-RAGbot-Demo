import datetime

import numpy as np
import pytest

from evaluation.embed_corpus import distinct_titles, paragraph_order_digest, title_order_digest
from evaluation.index_corpus import (
    IndexingError,
    build_actions,
    check_manifest,
    check_prefix,
    mapped_field,
    vector_field,
    verify_index,
)

MODEL = "Fake_Model_v1.0"
NOW = datetime.datetime(2026, 1, 1, 12, 0, 0)


def index_for(doc_id):
    return f"test_{round(doc_id / 1000)}"


@pytest.fixture
def records():
    return [
        {"doc_id": 1001, "title": "חופשה שנתית", "content": "פסקה א", "link": "https://e.org/1"},
        {"doc_id": 1001, "title": "חופשה שנתית", "content": "פסקה ב", "link": "https://e.org/1"},
        {"doc_id": 2600, "title": "דמי לידה", "content": "פסקה ג", "link": "https://e.org/2"},
    ]


@pytest.fixture
def vectors(records):
    paragraphs = np.arange(len(records) * 4, dtype=np.float32).reshape(len(records), 4)
    titles = distinct_titles([r["title"] for r in records])
    title_vectors = -np.arange(len(titles) * 4, dtype=np.float32).reshape(len(titles), 4)
    title_index = {t: i for i, t in enumerate(titles)}
    return paragraphs, title_vectors, title_index


def manifest_for(records, model=MODEL, corpus_sha256="abc"):
    titles = distinct_titles([r["title"] for r in records])
    return {
        "model": {"name": model},
        "corpus": {"sha256": corpus_sha256, "limit": None},
        "paragraphs": {
            "count": len(records),
            "order_sha256": paragraph_order_digest(
                [r["doc_id"] for r in records], [r["content"] for r in records]
            ),
        },
        "titles": {"count": len(titles), "order_sha256": title_order_digest(titles)},
    }


class FakeIndices:
    def __init__(self, es):
        self.es = es

    def refresh(self, index):
        pass

    def get_mapping(self, index):
        properties = {}
        for field in (vector_field("content", MODEL), vector_field("title", MODEL)):
            parent, child = field.split(".")
            properties[parent] = {
                "properties": {child: {"type": "dense_vector", "dims": self.es.dims}}
            }
        return {name: {"mappings": {"properties": properties}} for name in self.es.docs}


class FakeElasticsearch:
    def __init__(self, actions, dims):
        self.dims = dims
        self.docs = {}
        for action in actions:
            self.docs.setdefault(action["_index"], {})[action["_id"]] = action["_source"]
        self.indices = FakeIndices(self)

    def count(self, index, query=None):
        sources = [s for docs in self.docs.values() for s in docs.values()]
        if query is not None:
            field = query["exists"]["field"]
            sources = [s for s in sources if field in s]
        return {"count": len(sources)}

    def get(self, index, id):
        return {"_source": self.docs[index][id]}


def test_build_actions_use_engine_shape_and_index_names(records, vectors):
    paragraphs, title_vectors, title_index = vectors

    actions = list(
        build_actions(records, paragraphs, title_vectors, title_index, index_for, MODEL, NOW)
    )

    assert [a["_index"] for a in actions] == ["test_1", "test_1", "test_3"]
    assert [a["_id"] for a in actions] == ["0", "1", "2"]
    source = actions[2]["_source"]
    assert set(source) == {
        "last_update",
        "doc_id",
        "title",
        "content",
        "link",
        "content_Fake_Model_v1.0_vectors",
        "title_Fake_Model_v1.0_vectors",
    }
    assert source["last_update"] == NOW
    assert source["doc_id"] == 2600
    assert source["content_Fake_Model_v1.0_vectors"] == paragraphs[2].tolist()
    assert source["title_Fake_Model_v1.0_vectors"] == title_vectors[1].tolist()
    assert actions[0]["_source"]["title_Fake_Model_v1.0_vectors"] == title_vectors[0].tolist()


def test_check_manifest_accepts_matching_embeddings(records):
    check_manifest(manifest_for(records), records, "abc", MODEL)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: (manifest_for(list(reversed(r))), r, "abc", MODEL), "paragraph order"),
        (lambda r: (manifest_for(r[:2]), r, "abc", MODEL), "paragraph count"),
        (lambda r: (manifest_for(r), r, "other", MODEL), "corpus SHA-256"),
        (lambda r: (manifest_for(r, model="Other"), r, "abc", MODEL), "model"),
    ],
)
def test_check_manifest_refuses_mismatches(records, change, message):
    with pytest.raises(IndexingError, match=message):
        check_manifest(*change(records))


def test_check_prefix_accepts_own_indices():
    check_prefix(["kolzchut_full_0", "kolzchut_full_22"], "kolzchut_full")
    check_prefix([], "kolzchut_full")


@pytest.mark.parametrize(
    "names, prefix",
    [
        (["kolzchut_paragraphs_0"], "kolzchut"),
        (["kolzchut_full_0", "kolzchut_full_backup"], "kolzchut_full"),
    ],
)
def test_check_prefix_refuses_foreign_indices(names, prefix):
    with pytest.raises(IndexingError):
        check_prefix(names, prefix)


def test_verify_index_passes_on_a_complete_index(records, vectors):
    paragraphs, title_vectors, title_index = vectors
    actions = build_actions(records, paragraphs, title_vectors, title_index, index_for, MODEL, NOW)
    es = FakeElasticsearch(actions, dims=4)

    result = verify_index(
        es, "test", records, paragraphs, title_vectors, title_index, index_for, MODEL
    )

    assert result["ok"] is True
    assert result["indices"] == ["test_1", "test_3"]
    assert result["counts"]["documents"] == 3


def test_verify_index_detects_missing_and_wrong_vectors(records, vectors):
    paragraphs, title_vectors, title_index = vectors
    actions = build_actions(records, paragraphs, title_vectors, title_index, index_for, MODEL, NOW)
    es = FakeElasticsearch(actions, dims=4)
    es.docs["test_1"]["1"]["content_Fake_Model_v1.0_vectors"] = [0.0, 0.0, 0.0, 0.0]
    del es.docs["test_3"]["2"]["title_Fake_Model_v1.0_vectors"]

    result = verify_index(
        es, "test", records[:2], paragraphs, title_vectors, title_index, index_for, MODEL
    )

    assert result["ok"] is False
    assert result["checks"]["document_count"] is False
    assert result["checks"]["sampled_vectors"] is False


def test_mapped_field_follows_dots_as_object_paths():
    mapping = {
        "properties": {
            "doc_id": {"type": "long"},
            "content_M_v1": {"properties": {"0_vectors": {"type": "dense_vector", "dims": 4}}},
        }
    }

    assert mapped_field(mapping, "content_M_v1.0_vectors") == {"type": "dense_vector", "dims": 4}
    assert mapped_field(mapping, "doc_id") == {"type": "long"}
    assert mapped_field(mapping, "title_M_v1.0_vectors") == {}
