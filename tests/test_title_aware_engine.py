import logging
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from evaluation.index_corpus import vector_field

APP_SRC = Path(__file__).resolve().parents[1] / "app" / "src"
sys.path.insert(0, str(APP_SRC))

# The engine reads its document definitions when it is first imported.
with patch.dict(os.environ, {"DOCUMENT_DEFINITION_CONFIG": str(APP_SRC / "doc-config.json")}):
    from retrieval.title_aware_engine import (
        TitleAwareEngine,
        rank_documents,
        title_vector_field,
    )
    from webiks_hebrew_ragbot.engine import Engine

QUERY = np.array([1.0, 0.0], dtype=np.float32)


def at_cosine(c):
    """A unit vector whose cosine with QUERY is c."""
    return [c, math.sqrt(1.0 - c * c)]


class FakeModel:
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def eval(self):
        return self

    def encode(self, texts):
        self.calls.append(texts)
        if isinstance(texts, str):
            return np.asarray(self.vectors.get(texts, QUERY), dtype=np.float32)
        return np.asarray([self.vectors[t] for t in texts], dtype=np.float32)


class FakeElasticModel:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.created = []
        self.updated = []

    def search(self, embedded_search, size=50):
        return self.hits

    def create_paragraph(self, paragraph):
        self.created.append(paragraph)

    def create_or_update_documents(self, paragraphs_dicts, update=False):
        self.updated.append((paragraphs_dicts, update))


def make_engine(hits=(), weight=0.2, vectors=None, cls=TitleAwareEngine):
    kwargs = {"elastic_model": FakeElasticModel(hits), "retrieval_model": FakeModel(vectors)}
    if cls is TitleAwareEngine:
        kwargs["title_weight"] = weight
    return cls(llms_client=None, **kwargs)


TITLE_FIELD = title_vector_field(make_engine().model_name)


def hit(doc_id, paragraph, title=None, content=None):
    source = {"doc_id": doc_id, "title": f"title {doc_id}", "content": content or f"p{doc_id}"}
    if title is not None:
        source[TITLE_FIELD] = at_cosine(title)
    return {"_score": paragraph + 1.0, "_source": source}


def ids(documents):
    return [d.doc_id for d in documents]


def without_title(source):
    return {key: value for key, value in source.items() if key != TITLE_FIELD}


# Ties (docs 2 and 3), a second paragraph of doc 1, and titles that would reorder at w > 0.
ENGINE_ORDER_HITS = [
    hit(1, 0.90, title=0.0),
    hit(2, 0.80, title=0.9),
    hit(3, 0.80, title=0.5),
    hit(1, 0.70, title=0.0),
    hit(4, 0.60, title=1.0),
    hit(2, 0.50, title=0.9),
]


def test_zero_weight_keeps_the_engine_order():
    ranked = rank_documents(ENGINE_ORDER_HITS, QUERY, 0.0, TITLE_FIELD)

    assert ids(ranked) == [1, 2, 3, 4]
    assert [d.score for d in ranked] == [d.paragraph_score for d in ranked]


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_zero_weight_search_equals_the_engine_search(top_k):
    base = make_engine(ENGINE_ORDER_HITS, cls=Engine)
    title_aware = make_engine(ENGINE_ORDER_HITS, weight=0.0)

    expected = [without_title(source) for source in base.search_documents("q", top_k)]
    assert title_aware.search_documents("q", top_k) == expected


def test_title_term_reorders_documents():
    hits = [hit(1, 0.80, title=0.10), hit(2, 0.75, title=0.90)]

    ranked = rank_documents(hits, QUERY, 0.2, TITLE_FIELD)

    assert ids(ranked) == [2, 1]
    assert ranked[0].score == pytest.approx(0.93)
    assert ranked[1].score == pytest.approx(0.82)
    assert ranked[0].title_score == pytest.approx(0.90)
    assert ranked[0].paragraph_score == pytest.approx(0.75)


def test_best_paragraph_represents_the_document():
    hits = [hit(1, 0.90, title=0.2, content="best"), hit(2, 0.85), hit(1, 0.50, content="other")]

    ranked = rank_documents(hits, QUERY, 0.5, TITLE_FIELD)

    assert ranked[0].doc_id == 1
    assert ranked[0].paragraph_score == pytest.approx(0.90)
    assert ranked[0].source["content"] == "best"


def test_equal_scores_keep_the_engine_order():
    first = [hit(7, 0.70, title=0.5), hit(3, 0.70, title=0.5)]

    assert ids(rank_documents(first, QUERY, 0.3, TITLE_FIELD)) == [7, 3]
    assert ids(rank_documents(first[::-1], QUERY, 0.3, TITLE_FIELD)) == [3, 7]


def test_title_vector_may_come_from_another_paragraph_of_the_document():
    hits = [hit(1, 0.80), hit(1, 0.60, title=0.5)]

    ranked = rank_documents(hits, QUERY, 0.2, TITLE_FIELD)

    assert ranked[0].title_score == pytest.approx(0.5)
    assert ranked[0].score == pytest.approx(0.90)


def test_document_without_title_vector_gets_no_title_term():
    hits = [hit(1, 0.80), hit(2, 0.75, title=0.90)]

    ranked = rank_documents(hits, QUERY, 0.2, TITLE_FIELD)

    assert ids(ranked) == [2, 1]
    assert ranked[1].title_score is None
    assert ranked[1].score == pytest.approx(0.80)


def test_missing_title_vectors_are_logged_once(caplog):
    engine = make_engine([hit(1, 0.80), hit(2, 0.75, title=0.90)])

    with caplog.at_level(logging.WARNING, logger="retrieval.title_aware_engine"):
        engine.search_documents("q", 3)
        engine.search_documents("q", 3)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert TITLE_FIELD in warnings[0].getMessage()


def test_no_warning_when_every_document_has_a_title_vector(caplog):
    engine = make_engine([hit(1, 0.80, title=0.1), hit(2, 0.75, title=0.9)])

    with caplog.at_level(logging.WARNING, logger="retrieval.title_aware_engine"):
        engine.search_documents("q", 3)

    assert not caplog.records


def test_fewer_documents_than_top_k_returns_them_all():
    engine = make_engine([hit(1, 0.80, title=0.1), hit(2, 0.75, title=0.9), hit(1, 0.7)])

    documents = engine.search_documents("q", 10)

    assert [d["doc_id"] for d in documents] == [2, 1]


def test_search_returns_the_best_paragraph_sources():
    hits = [hit(1, 0.80, title=0.1, content="a"), hit(2, 0.75, title=0.9, content="b")]
    engine = make_engine(hits)

    assert engine.search_documents("q", 1) == [without_title(hits[1]["_source"])]


def test_search_returns_no_title_vectors_and_leaves_the_hits_unchanged():
    hits = [hit(1, 0.80, title=0.1), hit(2, 0.75, title=0.9), hit(3, 0.70)]
    engine = make_engine(hits)

    documents = engine.search_documents("q", 3)

    assert [d["doc_id"] for d in documents] == [2, 1, 3]
    assert all(TITLE_FIELD not in d for d in documents)
    assert all(TITLE_FIELD in h["_source"] for h in hits[:2])


PARAGRAPHS_VECTORS = {
    "חופשה שנתית": [0.0, 1.0],
    "דמי לידה": [1.0, 0.0],
    "פסקה א": [0.5, 0.75],
    "פסקה ב": [0.75, 0.5],
    "פסקה ג": [0.0, 1.0],
    "פסקה ד": [1.0, 0.0],
}


def paragraphs():
    return [
        {"doc_id": 1001, "title": "חופשה שנתית", "content": "פסקה א", "link": "l1"},
        {"doc_id": 1001, "title": "חופשה שנתית", "content": "פסקה ב", "link": "l1"},
        {"doc_id": 2600, "title": "דמי לידה", "content": "פסקה ג", "link": "l2"},
        {"doc_id": 2600, "title": "דמי לידה", "content": "פסקה ד", "link": "l2"},
    ]


def content_field(engine):
    return f"content_{engine.model_name}_vectors"


def test_create_paragraphs_encodes_each_title_once():
    engine = make_engine(vectors=PARAGRAPHS_VECTORS)

    engine.create_paragraphs(paragraphs())

    title_calls = [c for c in engine.retrieval_model.calls if not isinstance(c, str)]
    assert title_calls == [["חופשה שנתית", "דמי לידה"]]
    created = engine.elastic_model.created
    assert len(created) == 4
    for doc in created:
        np.testing.assert_array_equal(doc[TITLE_FIELD], PARAGRAPHS_VECTORS[doc["title"]])
        np.testing.assert_array_equal(
            doc[content_field(engine)], PARAGRAPHS_VECTORS[doc["content"]]
        )


@pytest.mark.parametrize("delete_existing", [False, True])
def test_update_docs_adds_title_vectors(delete_existing):
    engine = make_engine(vectors=PARAGRAPHS_VECTORS)

    engine.update_docs(paragraphs()[:2], delete_existing)

    title_calls = [c for c in engine.retrieval_model.calls if not isinstance(c, str)]
    assert title_calls == [["חופשה שנתית"]]
    [(docs, update)] = engine.elastic_model.updated
    assert update is delete_existing
    for doc in docs:
        np.testing.assert_array_equal(doc[TITLE_FIELD], PARAGRAPHS_VECTORS["חופשה שנתית"])
        assert content_field(engine) in doc


def test_paragraphs_without_title_are_indexed_without_title_vector():
    engine = make_engine(vectors=PARAGRAPHS_VECTORS)
    docs = [{"doc_id": 5, "title": "", "content": "פסקה א", "link": "l"}]

    engine.create_paragraphs(docs)

    assert all(isinstance(c, str) for c in engine.retrieval_model.calls)
    assert TITLE_FIELD not in engine.elastic_model.created[0]


def test_title_field_matches_the_corpus_indexer():
    model = make_engine().model_name

    assert title_vector_field(model) == vector_field("title", model)
