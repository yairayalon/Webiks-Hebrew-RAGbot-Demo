"""Title-aware retrieval on top of the ``webiks-hebrew-ragbot`` engine.

Paragraphs are sections of a page and rarely name the page, so paragraphs of sibling pages
look alike to the embedding model. The page title tells them apart. Search runs in two
stages:

1. The engine's dense search: the 50 paragraphs most similar to the query, and the best
   paragraph of each document among them.
2. Each of those documents is re-scored as
   ``cos(query, best paragraph) + weight * cos(query, title)``.

Title vectors come from the same model. They are stored on every paragraph, next to the
content vector, so every existing create, update and delete path keeps them in sync.
With weight 0 the ranking is exactly the engine's.
"""

import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from webiks_hebrew_ragbot.engine import Engine
from webiks_hebrew_ragbot.llm_client import LLMClient

logger = logging.getLogger(__name__)

TITLE_FIELD = "title"


def title_vector_field(model_name: str) -> str:
    """Return the name of the title vector field, mirroring the engine's content field.

    Args:
        model_name (str): Embedding model name.

    Returns:
        str: ``title_<model_name>_vectors``.
    """
    return f"{TITLE_FIELD}_{model_name}_vectors"


@dataclass(frozen=True)
class RankedDocument:
    """A candidate document and its scores.

    Attributes:
        doc_id: The document's identifier, as stored.
        score (float): Combined score used for ranking.
        paragraph_score (float): Cosine similarity of the query and the best paragraph.
        title_score (float | None): Cosine similarity of the query and the title; None
            when the document has no title vector.
        source (dict): The best paragraph's ``_source``, as the engine returns it.
    """

    doc_id: Hashable
    score: float
    paragraph_score: float
    title_score: float | None
    source: dict


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors, computed in float64.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_documents(
    hits: Sequence[dict],
    query_vector: np.ndarray,
    weight: float,
    title_field: str,
    identifier: str = "doc_id",
) -> list[RankedDocument]:
    """Rank the documents found by the engine's paragraph search.

    Args:
        hits (Sequence[dict]): Elasticsearch hits in the engine's order, best first; each
            ``_score`` is the engine's cosine similarity + 1.
        query_vector (np.ndarray): The query embedding.
        weight (float): Weight of the query-title similarity.
        title_field (str): Name of the title vector field in ``_source``.
        identifier (str): Document identifier field.

    Returns:
        list[RankedDocument]: Every document among the hits, best first. Equal scores keep
            the engine's order, so weight 0 gives exactly the engine's ranking.
    """
    first_hits: dict = {}
    title_vectors: dict = {}
    for hit in hits:
        source = hit["_source"]
        doc_id = source[identifier]
        first_hits.setdefault(doc_id, hit)
        if doc_id not in title_vectors and source.get(title_field) is not None:
            title_vectors[doc_id] = source[title_field]

    documents = []
    for doc_id, hit in first_hits.items():
        paragraph_score = hit["_score"] - 1.0
        title_vector = title_vectors.get(doc_id)
        title_score = None if title_vector is None else cosine(query_vector, title_vector)
        score = paragraph_score if title_score is None else paragraph_score + weight * title_score
        documents.append(
            RankedDocument(doc_id, score, paragraph_score, title_score, hit["_source"])
        )
    return sorted(documents, key=lambda document: -document.score)


class TitleAwareEngine(Engine):
    """The engine with title vectors at indexing time and title re-scoring at search time.

    Attributes:
        title_weight (float): Weight of the query-title similarity.
        title_field (str): Name of the title vector field.
    """

    def __init__(self, llms_client: LLMClient | None, title_weight: float, **engine_kwargs):
        """Initialize the engine.

        Args:
            llms_client (LLMClient | None): The LLM client, as for ``Engine``; None when
                only searching.
            title_weight (float): Weight of the query-title similarity.
            **engine_kwargs: Other ``Engine`` arguments (``es_client``, ``retrieval_model``...).
        """
        super().__init__(llms_client=llms_client, **engine_kwargs)
        self.title_weight = title_weight
        self.title_field = title_vector_field(self.model_name)
        self._warned_missing_titles = False

    def add_title_vectors(self, docs: list[dict]) -> None:
        """Add a title vector to every paragraph that has a title.

        Each distinct title is encoded once, in a single batch.

        Args:
            docs (list[dict]): Paragraphs to be indexed; updated in place.
        """
        titles = list(dict.fromkeys(doc[TITLE_FIELD] for doc in docs if doc.get(TITLE_FIELD)))
        if not titles:
            return
        vectors = dict(zip(titles, self.retrieval_model.encode(titles), strict=True))
        for doc in docs:
            if doc.get(TITLE_FIELD):
                doc[self.title_field] = vectors[doc[TITLE_FIELD]]

    def create_paragraphs(self, list_of_paragraphs: list[dict]) -> None:
        """Embed paragraphs and their titles and add them to Elasticsearch.

        Args:
            list_of_paragraphs (list[dict]): Paragraphs to index.
        """
        self.add_title_vectors(list_of_paragraphs)
        super().create_paragraphs(list_of_paragraphs)

    def update_docs(self, list_of_docs: list[dict], delete_existing: bool = False) -> None:
        """Create or update documents, with title vectors.

        Args:
            list_of_docs (list[dict]): Paragraphs of one document.
            delete_existing (bool): Whether to delete the document's existing paragraphs.
        """
        self.add_title_vectors(list_of_docs)
        super().update_docs(list_of_docs, delete_existing)

    def search_documents(self, query: str, top_k: int) -> list[dict]:
        """Search for documents and re-score them with their titles.

        Args:
            query (str): The query.
            top_k (int): Number of documents to return.

        Returns:
            list[dict]: The top ``top_k`` documents, as the best paragraph's ``_source`` each,
                without the title vector: only the ranking uses it, and the documents go on to
                the LLM client (the mock client copies them into its answer).
        """
        query_vector = self.retrieval_model.encode(query)
        hits = self.elastic_model.search(query_vector)
        ranked = rank_documents(
            hits, query_vector, self.title_weight, self.title_field, self.identifier_field
        )
        if not self._warned_missing_titles and any(d.title_score is None for d in ranked):
            self._warned_missing_titles = True
            logger.warning(
                "Some documents have no %s; they are ranked without a title term. "
                "Re-index with title-aware retrieval on to add title vectors.",
                self.title_field,
            )
        return [
            {key: value for key, value in document.source.items() if key != self.title_field}
            for document in ranked[:top_k]
        ]
