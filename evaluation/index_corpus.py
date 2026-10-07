"""Index the full paragraph corpus in Elasticsearch from precomputed embeddings.

Indexing through the Demo's ``/initialize_elastic_from_json`` embeds every paragraph on the
fly, which takes about a day on a laptop CPU. This script writes the documents the engine
would write -- same fields, same index names (``index_from_doc_id``), same dynamic mapping --
using the vectors from ``embed_corpus.py``, plus each paragraph's title vector.

Usage:
    python -m evaluation.index_corpus --embeddings-dir DIR [--index-prefix PREFIX] [--recreate]
    python -m evaluation.index_corpus --embeddings-dir DIR --verify-only
"""

import argparse
import datetime
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

from evaluation.embed_corpus import (
    MANIFEST_FILE,
    distinct_titles,
    paragraph_order_digest,
    title_order_digest,
)
from evaluation.engine_setup import (
    DEFAULT_CORPUS,
    configure_engine,
    file_sha256,
    load_corpus_records,
    model_name,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PREFIX = "kolzchut_full"
VERIFY_SAMPLE = 20
SAMPLE_SEED = 0


class IndexingError(RuntimeError):
    """Raised when the embeddings, corpus or existing indices don't fit together."""


def vector_field(field: str, model: str) -> str:
    """Return the engine's name for a vector field.

    Args:
        field (str): Source field, e.g. ``content`` or ``title``.
        model (str): Model name.

    Returns:
        str: ``<field>_<model>_vectors``.
    """
    return f"{field}_{model}_vectors"


def mapped_field(mapping: dict, path: str) -> dict:
    """Find a field in an index mapping.

    Elasticsearch maps dots in field names as object paths, so the engine's
    ``content_<model>_v1.0_vectors`` lives at ``content_<model>_v1`` → ``0_vectors``.

    Args:
        mapping (dict): The index's ``mappings`` object.
        path (str): Field name as written in the documents.

    Returns:
        dict: The field's mapping, or an empty dict if it isn't mapped.
    """
    node = mapping
    for part in path.split("."):
        node = node.get("properties", {}).get(part)
        if node is None:
            return {}
    return node


def check_manifest(manifest: dict, records: Sequence[dict], corpus_sha256: str, model: str) -> None:
    """Make sure the embeddings belong to this corpus, in this order, and to this model.

    Args:
        manifest (dict): The embeddings manifest.
        records (Sequence[dict]): Corpus records, truncated to the manifest's limit.
        corpus_sha256 (str): SHA-256 of the local corpus file.
        model (str): Model name from the Demo's document configuration.

    Raises:
        IndexingError: On any mismatch.
    """
    titles = distinct_titles([r["title"] for r in records])
    problems = []
    if manifest["corpus"]["sha256"] != corpus_sha256:
        problems.append("corpus SHA-256")
    if manifest["model"]["name"] != model:
        problems.append(f"model ({manifest['model']['name']} vs {model})")
    if manifest["paragraphs"]["count"] != len(records):
        problems.append("paragraph count")
    elif manifest["paragraphs"]["order_sha256"] != paragraph_order_digest(
        [r["doc_id"] for r in records], [r["content"] for r in records]
    ):
        problems.append("paragraph order")
    if manifest["titles"]["order_sha256"] != title_order_digest(titles):
        problems.append("title order")
    if problems:
        raise IndexingError(f"The embeddings don't match the corpus: {', '.join(problems)}")


def check_prefix(names: Sequence[str], prefix: str) -> None:
    """Refuse to touch indices that merely share the prefix.

    Args:
        names (Sequence[str]): Existing indices matching ``<prefix>*``.
        prefix (str): Index prefix.

    Raises:
        IndexingError: If an index isn't named ``<prefix>_<number>``.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_\d+")
    foreign = [name for name in names if not pattern.fullmatch(name)]
    if foreign:
        raise IndexingError(f"Indices {foreign} share the prefix {prefix!r}; choose another prefix")


def build_actions(
    records: Sequence[dict],
    paragraphs: np.ndarray,
    title_vectors: np.ndarray,
    title_index: dict[str, int],
    index_for: Callable[[int], str],
    model: str,
    now: datetime.datetime,
) -> Iterator[dict]:
    """Yield one bulk action per paragraph, in the engine's document shape.

    The ``_id`` is the paragraph's corpus position, so indexing again overwrites rather
    than duplicates.

    Args:
        records (Sequence[dict]): Corpus records (``doc_id``, ``title``, ``content``, ``link``).
        paragraphs (np.ndarray): Paragraph vectors, one row per record.
        title_vectors (np.ndarray): Title vectors.
        title_index (dict[str, int]): Row of each title in ``title_vectors``.
        index_for (Callable[[int], str]): Index name for a document id.
        model (str): Model name, used in the vector field names.
        now (datetime.datetime): ``last_update`` value.

    Yields:
        dict: A bulk ``index`` action.
    """
    content_field = vector_field("content", model)
    title_field = vector_field("title", model)
    for position, record in enumerate(records):
        doc_id = int(record["doc_id"])
        source = {
            "last_update": now,
            **record,
            "doc_id": doc_id,
            content_field: paragraphs[position].tolist(),
            title_field: title_vectors[title_index[record["title"]]].tolist(),
        }
        yield {
            "_op_type": "index",
            "_index": index_for(doc_id),
            "_id": str(position),
            "_source": source,
        }


def existing_indices(es, prefix: str) -> list[str]:
    """List the indices whose names start with the prefix.

    Args:
        es: Elasticsearch client.
        prefix (str): Index prefix.

    Returns:
        list[str]: Sorted index names.
    """
    return sorted(es.indices.get(index=f"{prefix}*"))


def bulk_index(es, actions: Iterator[dict], total: int, chunk_size: int) -> None:
    """Send the actions in chunks, logging progress.

    Args:
        es: Elasticsearch client.
        actions (Iterator[dict]): Bulk actions.
        total (int): Number of actions, for progress logs.
        chunk_size (int): Documents per bulk request.
    """
    from elasticsearch import helpers

    start = time.perf_counter()
    done = 0
    for _ in helpers.streaming_bulk(
        es, actions, chunk_size=chunk_size, max_retries=3, raise_on_error=True
    ):
        done += 1
        if done % 2000 == 0 or done == total:
            logger.info("Indexed %d/%d in %.0f s", done, total, time.perf_counter() - start)


def verify_index(
    es,
    prefix: str,
    records: Sequence[dict],
    paragraphs: np.ndarray,
    title_vectors: np.ndarray,
    title_index: dict[str, int],
    index_for: Callable[[int], str],
    model: str,
) -> dict:
    """Check document counts, vector mappings and a sample of stored vectors.

    Args:
        es: Elasticsearch client.
        prefix (str): Index prefix.
        records (Sequence[dict]): Corpus records.
        paragraphs (np.ndarray): Paragraph vectors.
        title_vectors (np.ndarray): Title vectors.
        title_index (dict[str, int]): Row of each title in ``title_vectors``.
        index_for (Callable[[int], str]): Index name for a document id.
        model (str): Model name.

    Returns:
        dict: Indices, counts, each check's outcome and ``ok``.
    """
    pattern = f"{prefix}_*"
    es.indices.refresh(index=pattern)
    fields = [vector_field("content", model), vector_field("title", model)]
    expected = len(records)
    counts = {"documents": es.count(index=pattern)["count"]}
    for field in fields:
        counts[field] = es.count(index=pattern, query={"exists": {"field": field}})["count"]

    dim = paragraphs.shape[1]
    mappings = es.indices.get_mapping(index=pattern)
    mapping_ok = all(
        (spec := mapped_field(mapping["mappings"], field)).get("type") == "dense_vector"
        and spec.get("dims") == dim
        for mapping in mappings.values()
        for field in fields
    )

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = rng.choice(expected, size=min(VERIFY_SAMPLE, expected), replace=False)
    vectors_ok = True
    for position in sample:
        record = records[position]
        source = es.get(index=index_for(int(record["doc_id"])), id=str(position))["_source"]
        stored_content = np.asarray(source[fields[0]], dtype=np.float32)
        stored_title = np.asarray(source[fields[1]], dtype=np.float32)
        vectors_ok &= np.array_equal(stored_content, paragraphs[position])
        vectors_ok &= np.array_equal(stored_title, title_vectors[title_index[record["title"]]])
        vectors_ok &= int(source["doc_id"]) == int(record["doc_id"])

    checks = {
        "document_count": counts["documents"] == expected,
        "vector_counts": all(counts[field] == expected for field in fields),
        "dense_vector_mapping": mapping_ok,
        "sampled_vectors": bool(vectors_ok),
    }
    return {
        "indices": sorted(mappings),
        "expected_documents": expected,
        "counts": counts,
        "checks": checks,
        "ok": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """Index the corpus, then verify the index.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Exit code; 1 when a check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--recreate", action="store_true", help="delete existing indices first")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    # The engine logs every search on the root logger; show only this package's progress.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("evaluation", __name__):
        logging.getLogger(name).setLevel(logging.INFO)

    configure_engine(args.index_prefix)
    from elasticsearch import Elasticsearch
    from webiks_hebrew_ragbot.elastic_model import index_from_doc_id

    with open(args.embeddings_dir / MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    records = load_corpus_records(args.corpus)
    if manifest["corpus"]["limit"] is not None:
        records = records[: manifest["corpus"]["limit"]]
    model = model_name()
    paragraphs = np.load(args.embeddings_dir / manifest["paragraphs"]["file"])
    title_vectors = np.load(args.embeddings_dir / manifest["titles"]["file"])
    titles = distinct_titles([r["title"] for r in records])
    title_index = {title: row for row, title in enumerate(titles)}

    try:
        check_manifest(manifest, records, file_sha256(args.corpus), model)
        if paragraphs.shape[0] != len(records) or title_vectors.shape[0] != len(titles):
            raise IndexingError("The embedding files don't have one row per paragraph and title")
        es = Elasticsearch(args.es_url, request_timeout=300)
        names = existing_indices(es, args.index_prefix)
        check_prefix(names, args.index_prefix)
    except IndexingError as e:
        logger.error("%s", e)
        return 1

    if not args.verify_only:
        if names and args.recreate:
            for name in names:
                es.indices.delete(index=name)
            logger.info("Deleted %d indices: %s", len(names), ", ".join(names))
        elif names:
            logger.info("Indices exist; documents with the same ids are overwritten")
        now = datetime.datetime.now()
        actions = build_actions(
            records, paragraphs, title_vectors, title_index, index_from_doc_id, model, now
        )
        bulk_index(es, actions, len(records), args.chunk_size)

    result = verify_index(
        es,
        args.index_prefix,
        records,
        paragraphs,
        title_vectors,
        title_index,
        index_from_doc_id,
        model,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
