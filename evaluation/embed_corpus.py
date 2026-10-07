"""Embed the paragraph corpus and the page titles with the retrieval model, in resumable shards.

The script is self-contained so it can run on a GPU machine (e.g. Colab) as a single file:
it imports only numpy, pandas and the standard library at module level, and loads torch and
sentence-transformers when it needs the model.

Paragraph vectors follow the row order of ``pandas.read_json(corpus)``, the order the Demo
indexes in. Title vectors cover the distinct titles in order of first appearance. Each shard
is written atomically; a rerun skips the shards that load correctly, so an interrupted run
resumes where it stopped.

Usage:
    python embed_corpus.py --corpus CORPUS --model-dir MODEL_DIR --out-dir OUT [--batch-size 32]
    python embed_corpus.py --finalize --corpus CORPUS --model-dir MODEL_DIR --out-dir OUT
    python embed_corpus.py --verify-sample 20 --corpus CORPUS --model-dir MODEL_DIR --out-dir OUT
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARAGRAPHS = "paragraphs"
TITLES = "titles"
SHARDS_DIR = "shards"
RUN_FILE = "run.json"
MANIFEST_FILE = "manifest.json"
MIN_COSINE = 0.9999
NORM_TOLERANCE = 1e-3
SAMPLE_SEED = 0

Encoder = Callable[[Sequence[str]], np.ndarray]


class EmbeddingError(RuntimeError):
    """Raised when inputs, shards or outputs are inconsistent."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Kept here rather than imported, so this script runs on its own.

    Args:
        path (Path): File to hash.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraph_order_digest(doc_ids: Sequence[int], contents: Sequence[str]) -> str:
    """Hash the paragraph order, so vectors can be matched to corpus rows elsewhere.

    Args:
        doc_ids (Sequence[int]): Document id of each paragraph, in order.
        contents (Sequence[str]): Text of each paragraph, in the same order.

    Returns:
        str: SHA-256 hex digest of the JSON list of ``[doc_id, content]`` pairs.
    """
    pairs = [[int(d), str(c)] for d, c in zip(doc_ids, contents, strict=True)]
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()


def title_order_digest(titles: Sequence[str]) -> str:
    """Hash the ordered list of distinct titles.

    Args:
        titles (Sequence[str]): Distinct titles, in order.

    Returns:
        str: SHA-256 hex digest of the JSON list.
    """
    return hashlib.sha256(
        json.dumps([str(t) for t in titles], ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def distinct_titles(titles: Sequence[str]) -> list[str]:
    """Return the distinct titles in order of first appearance.

    Args:
        titles (Sequence[str]): The title of each paragraph.

    Returns:
        list[str]: Distinct titles.
    """
    return list(dict.fromkeys(str(t) for t in titles))


def load_corpus(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Read the paragraph corpus the way the Demo does (``pandas.read_json``).

    Args:
        path (Path): Corpus JSON.
        limit (int | None): Keep only the first ``limit`` paragraphs (smoke runs).

    Returns:
        pd.DataFrame: One row per paragraph.
    """
    df = pd.read_json(path)
    return df.iloc[:limit] if limit is not None else df


def shard_path(shards_dir: Path, kind: str, index: int) -> Path:
    """Return the path of one shard.

    Args:
        shards_dir (Path): Directory holding the shards.
        kind (str): ``paragraphs`` or ``titles``.
        index (int): Shard number.

    Returns:
        Path: The shard file.
    """
    return shards_dir / f"{kind}_{index:05d}.npy"


def save_atomic(path: Path, array: np.ndarray) -> None:
    """Write an array so that ``path`` either holds the complete file or does not exist.

    Args:
        path (Path): Destination ``.npy`` file.
        array (np.ndarray): Array to save.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON so that ``path`` either holds the complete file or keeps its old content.

    Args:
        path (Path): Destination file.
        data (dict): JSON-serializable content.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_valid_shard(path: Path, rows: int) -> np.ndarray | None:
    """Load a shard if it is complete and sane.

    A shard copied from a synced drive can be truncated after a disconnect, so it is
    checked before being trusted.

    Args:
        path (Path): Shard file.
        rows (int): Expected number of rows.

    Returns:
        np.ndarray | None: The shard, or None when it is missing or invalid.
    """
    if not path.is_file():
        return None
    try:
        array = np.load(path)
    except (OSError, ValueError, EOFError) as e:
        logger.warning("Shard %s is unreadable (%s); recomputing it", path.name, e)
        return None
    if (
        array.ndim != 2
        or array.shape[0] != rows
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        logger.warning("Shard %s has shape %s; recomputing it", path.name, array.shape)
        return None
    return array


def embed_in_shards(
    texts: Sequence[str], kind: str, shards_dir: Path, shard_size: int, encode: Encoder
) -> None:
    """Embed texts shard by shard, skipping shards that already exist and load correctly.

    Args:
        texts (Sequence[str]): Texts to embed, in order.
        kind (str): ``paragraphs`` or ``titles``.
        shards_dir (Path): Directory for the shards.
        shard_size (int): Texts per shard.
        encode (Encoder): Maps a list of texts to a 2-D float array.
    """
    total = math.ceil(len(texts) / shard_size)
    for index in range(total):
        chunk = list(texts[index * shard_size : (index + 1) * shard_size])
        path = shard_path(shards_dir, kind, index)
        if load_valid_shard(path, len(chunk)) is not None:
            logger.info("%s shard %d/%d exists; skipping", kind, index + 1, total)
            continue
        start = time.perf_counter()
        vectors = np.asarray(encode(chunk), dtype=np.float32)
        if vectors.shape[0] != len(chunk):
            raise EmbeddingError(f"encoder returned {vectors.shape[0]} rows for {len(chunk)} texts")
        save_atomic(path, vectors)
        logger.info(
            "%s shard %d/%d: %d texts in %.1f s",
            kind,
            index + 1,
            total,
            len(chunk),
            time.perf_counter() - start,
        )


def load_run_file(shards_dir: Path) -> dict | None:
    """Read the shard directory's run file.

    Args:
        shards_dir (Path): Directory holding the shards.

    Returns:
        dict | None: The run file, or None if absent.
    """
    path = shards_dir / RUN_FILE
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def library_versions() -> dict:
    """Return the versions of the libraries that affect the vectors.

    Returns:
        dict: Version strings; libraries that are not installed are omitted.
    """
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    versions["pandas"] = pd.__version__
    for name in ("torch", "transformers", "sentence_transformers"):
        module = sys.modules.get(name)
        if module is not None:
            versions[name] = module.__version__
    return versions


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format, to the second."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def embed_corpus(
    corpus_path: Path,
    model_dir: Path,
    out_dir: Path,
    encode: Encoder,
    *,
    shard_size: int,
    limit: int | None = None,
    run_info: dict | None = None,
) -> None:
    """Embed every paragraph and distinct title into shards under ``out_dir/shards``.

    Args:
        corpus_path (Path): Corpus JSON.
        model_dir (Path): Model directory; its name identifies the model.
        out_dir (Path): Output directory.
        encode (Encoder): Maps a list of texts to a 2-D float array.
        shard_size (int): Texts per shard.
        limit (int | None): Embed only the first ``limit`` paragraphs (smoke runs).
        run_info (dict | None): Device, batch size and versions of this run, for provenance.

    Raises:
        EmbeddingError: If the shard directory belongs to a different configuration.
    """
    df = load_corpus(corpus_path, limit)
    titles = distinct_titles(df["title"])
    config = {
        "model": Path(model_dir).name,
        "corpus_sha256": file_sha256(corpus_path),
        "shard_size": shard_size,
        "limit": limit,
        "paragraphs": len(df),
        "titles": len(titles),
    }
    shards_dir = Path(out_dir) / SHARDS_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    run_file = load_run_file(shards_dir) or {"config": config, "runs": []}
    if run_file["config"] != config:
        raise EmbeddingError(
            f"{shards_dir} holds shards for {run_file['config']}, not {config}; use a new --out-dir"
        )
    run_file["runs"].append({"started_utc": utc_now(), **(run_info or {})})
    write_json_atomic(shards_dir / RUN_FILE, run_file)

    logger.info("Embedding %d paragraphs and %d titles", len(df), len(titles))
    embed_in_shards([str(c) for c in df["content"]], PARAGRAPHS, shards_dir, shard_size, encode)
    embed_in_shards(titles, TITLES, shards_dir, shard_size, encode)
    logger.info("All shards are complete; run with --finalize")


def merge_shards(shards_dir: Path, kind: str, count: int, shard_size: int) -> np.ndarray:
    """Concatenate one kind of shard into a single array.

    Args:
        shards_dir (Path): Directory holding the shards.
        kind (str): ``paragraphs`` or ``titles``.
        count (int): Expected total rows.
        shard_size (int): Rows per shard.

    Returns:
        np.ndarray: All vectors, in order.

    Raises:
        EmbeddingError: If a shard is missing or invalid.
    """
    parts = []
    for index in range(math.ceil(count / shard_size)):
        rows = min(shard_size, count - index * shard_size)
        array = load_valid_shard(shard_path(shards_dir, kind, index), rows)
        if array is None:
            raise EmbeddingError(f"{kind} shard {index} is missing or invalid; rerun the embedding")
        parts.append(array)
    return np.concatenate(parts)


def finalize(corpus_path: Path, model_dir: Path, out_dir: Path) -> dict:
    """Merge the shards into ``paragraphs.npy`` and ``titles.npy`` and write the manifest.

    Args:
        corpus_path (Path): Corpus JSON.
        model_dir (Path): Model directory.
        out_dir (Path): Output directory that holds ``shards/``.

    Returns:
        dict: The manifest.

    Raises:
        EmbeddingError: If the shards don't match the corpus or are incomplete.
    """
    out_dir = Path(out_dir)
    shards_dir = out_dir / SHARDS_DIR
    run_file = load_run_file(shards_dir)
    if run_file is None:
        raise EmbeddingError(f"No {RUN_FILE} in {shards_dir}; run the embedding first")
    config = run_file["config"]
    corpus_sha256 = file_sha256(corpus_path)
    if config["corpus_sha256"] != corpus_sha256 or config["model"] != Path(model_dir).name:
        raise EmbeddingError("The shards were computed for a different corpus or model")

    df = load_corpus(corpus_path, config["limit"])
    titles = distinct_titles(df["title"])
    paragraphs = merge_shards(shards_dir, PARAGRAPHS, len(df), config["shard_size"])
    title_vectors = merge_shards(shards_dir, TITLES, len(titles), config["shard_size"])
    if paragraphs.shape[1] != title_vectors.shape[1]:
        raise EmbeddingError("Paragraph and title vectors have different dimensions")

    save_atomic(out_dir / f"{PARAGRAPHS}.npy", paragraphs)
    save_atomic(out_dir / f"{TITLES}.npy", title_vectors)
    norms = np.linalg.norm(np.concatenate([paragraphs, title_vectors]), axis=1)
    safetensors = Path(model_dir) / "model.safetensors"
    manifest = {
        "model": {
            "name": Path(model_dir).name,
            "safetensors_sha256": file_sha256(safetensors) if safetensors.is_file() else None,
        },
        "corpus": {
            "file": Path(corpus_path).name,
            "sha256": corpus_sha256,
            "limit": config["limit"],
        },
        PARAGRAPHS: {
            "file": f"{PARAGRAPHS}.npy",
            "count": len(df),
            "documents": int(df["doc_id"].nunique()),
            "order": "rows of pandas.read_json(corpus), the order the Demo indexes",
            "order_sha256": paragraph_order_digest(df["doc_id"], df["content"]),
        },
        TITLES: {
            "file": f"{TITLES}.npy",
            "count": len(titles),
            "order": "distinct titles in order of first appearance",
            "order_sha256": title_order_digest(titles),
        },
        "dim": int(paragraphs.shape[1]),
        "dtype": str(paragraphs.dtype),
        "max_norm_deviation": float(np.abs(norms - 1.0).max()),
        "shard_size": config["shard_size"],
        "runs": run_file["runs"],
        "script_sha256": file_sha256(Path(__file__)),
        "created_utc": utc_now(),
    }
    write_json_atomic(out_dir / MANIFEST_FILE, manifest)
    logger.info(
        "Wrote %s %s and %s %s",
        f"{PARAGRAPHS}.npy",
        paragraphs.shape,
        f"{TITLES}.npy",
        title_vectors.shape,
    )
    return manifest


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity.

    Args:
        a (np.ndarray): First matrix.
        b (np.ndarray): Second matrix, same shape.

    Returns:
        np.ndarray: One cosine per row.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def verify(
    corpus_path: Path, model_dir: Path, out_dir: Path, sample_size: int, encode: Encoder
) -> dict:
    """Check finalized embeddings against the local corpus, model and a fresh sample.

    Args:
        corpus_path (Path): Local corpus JSON.
        model_dir (Path): Local model directory.
        out_dir (Path): Directory with ``paragraphs.npy``, ``titles.npy`` and the manifest.
        sample_size (int): Paragraphs to re-embed; half as many titles (at least one).
        encode (Encoder): Local encoder.

    Returns:
        dict: Each check's outcome and ``ok``.
    """
    out_dir = Path(out_dir)
    with open(out_dir / MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    paragraphs = np.load(out_dir / manifest[PARAGRAPHS]["file"])
    titles_vectors = np.load(out_dir / manifest[TITLES]["file"])
    df = load_corpus(corpus_path, manifest["corpus"]["limit"])
    titles = distinct_titles(df["title"])
    safetensors = Path(model_dir) / "model.safetensors"
    local_model_sha = file_sha256(safetensors) if safetensors.is_file() else None
    dim = manifest["dim"]

    checks = {
        "corpus_sha256": manifest["corpus"]["sha256"] == file_sha256(corpus_path),
        "model_sha256": manifest["model"]["safetensors_sha256"] == local_model_sha,
        "paragraph_order": manifest[PARAGRAPHS]["order_sha256"]
        == paragraph_order_digest(df["doc_id"], df["content"]),
        "title_order": manifest[TITLES]["order_sha256"] == title_order_digest(titles),
        "shapes": paragraphs.shape == (len(df), dim) and titles_vectors.shape == (len(titles), dim),
        "dtype": paragraphs.dtype == np.float32 and titles_vectors.dtype == np.float32,
        "finite": bool(np.isfinite(paragraphs).all() and np.isfinite(titles_vectors).all()),
    }
    norms = np.linalg.norm(np.concatenate([paragraphs, titles_vectors]), axis=1)
    checks["unit_norm"] = bool(np.abs(norms - 1.0).max() <= NORM_TOLERANCE)

    rng = np.random.default_rng(SAMPLE_SEED)
    paragraph_rows = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    title_rows = rng.choice(
        len(titles), size=min(max(1, sample_size // 2), len(titles)), replace=False
    )
    contents = [str(c) for c in df["content"]]
    paragraph_cos = cosines(
        np.asarray(encode([contents[i] for i in paragraph_rows]), dtype=np.float32),
        paragraphs[paragraph_rows],
    )
    title_cos = cosines(
        np.asarray(encode([titles[i] for i in title_rows]), dtype=np.float32),
        titles_vectors[title_rows],
    )
    min_cosine = float(min(paragraph_cos.min(), title_cos.min()))
    checks["sample_cosine"] = min_cosine >= MIN_COSINE

    return {
        "checks": checks,
        "sampled_paragraphs": len(paragraph_rows),
        "sampled_titles": len(title_rows),
        "min_cosine": min_cosine,
        "max_norm_deviation": float(np.abs(norms - 1.0).max()),
        "script_matches_manifest": manifest.get("script_sha256") == file_sha256(Path(__file__)),
        "ok": all(checks.values()),
    }


def load_encoder(
    model_dir: Path, device: str, batch_size: int, threads: int | None
) -> tuple[Encoder, dict]:
    """Load the sentence-transformers model and wrap its ``encode``.

    The engine calls ``encode`` with default arguments, so the model's own pooling and
    normalization are used unchanged here.

    Args:
        model_dir (Path): Model directory.
        device (str): ``auto``, ``cpu`` or ``cuda``.
        batch_size (int): Texts per forward pass.
        threads (int | None): CPU threads for torch; None keeps torch's default.

    Returns:
        tuple[Encoder, dict]: The encoder and this run's device, batch size and versions.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if threads:
        torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(model_dir), device=device)
    model.eval()
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = f"cpu ({torch.get_num_threads()} threads)"

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )

    run_info = {"device": device_name, "batch_size": batch_size, "versions": library_versions()}
    logger.info("Model %s loaded on %s", Path(model_dir).name, device_name)
    return encode, run_info


def main(argv: list[str] | None = None) -> int:
    """Embed, finalize or verify, depending on the arguments.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--finalize", action="store_true", help="merge the shards")
    mode.add_argument("--verify-sample", type=int, metavar="N", help="re-embed N paragraphs")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N paragraphs only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        if args.finalize:
            finalize(args.corpus, args.model_dir, args.out_dir)
            return 0
        encode, run_info = load_encoder(args.model_dir, args.device, args.batch_size, args.threads)
        if args.verify_sample is not None:
            result = verify(args.corpus, args.model_dir, args.out_dir, args.verify_sample, encode)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        embed_corpus(
            args.corpus,
            args.model_dir,
            args.out_dir,
            encode,
            shard_size=args.shard_size,
            limit=args.limit,
            run_info=run_info,
        )
        return 0
    except EmbeddingError as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
