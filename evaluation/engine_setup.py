"""Shared paths and engine setup for the evaluation scripts.

The engine (``webiks-hebrew-ragbot`` 1.0.1) reads its index prefix, model location and
document definitions from the environment when its modules are imported, and loads
``app/.env`` relative to the working directory without overriding variables that are
already set. The scripts therefore call ``configure_engine`` with absolute values before
importing it.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
APP_SRC = DEMO_ROOT / "app" / "src"
DOC_CONFIG = APP_SRC / "doc-config.json"
DEFAULT_MODEL_LOCATION = DEMO_ROOT / "app" / "artifacts"
DEFAULT_CORPUS = DEMO_ROOT / "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"
DEFAULT_QA_CSV = DEMO_ROOT / "Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv"
DEFAULT_REPORTS_DIR = DEMO_ROOT / "evaluation" / "reports"
ES_INDEX_LENGTH = "1000"
ENGINE_PACKAGE = "webiks_hebrew_ragbot"


def model_name() -> str:
    """Return the embedding model name configured in the Demo's ``doc-config.json``.

    Returns:
        str: The model directory name, e.g. ``Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0``.
    """
    with open(DOC_CONFIG, encoding="utf-8") as f:
        return json.load(f)["model_name"]


def add_app_src_to_path() -> None:
    """Make the Demo's ``app/src`` modules (e.g. ``utils``) importable."""
    app_src = str(APP_SRC)
    if app_src not in sys.path:
        sys.path.insert(0, app_src)


def configure_engine(index_prefix: str, model_location: Path = DEFAULT_MODEL_LOCATION) -> None:
    """Set the environment the engine reads at import time.

    Args:
        index_prefix (str): Elasticsearch index prefix (``ES_EMBEDDING_INDEX``).
        model_location (Path): Directory that contains the model directory.

    Raises:
        RuntimeError: If the engine was imported before, so these values would be ignored.
    """
    imported = [m for m in sys.modules if m.split(".")[0] == ENGINE_PACKAGE]
    if imported:
        raise RuntimeError(
            f"{ENGINE_PACKAGE} was imported before configure_engine(); "
            "it reads its settings at import time"
        )
    os.environ["ES_EMBEDDING_INDEX"] = index_prefix
    os.environ["ES_EMBEDDING_INDEX_LENGTH"] = ES_INDEX_LENGTH
    os.environ["MODEL_LOCATION"] = str(Path(model_location).resolve())
    os.environ["DOCUMENT_DEFINITION_CONFIG"] = str(DOC_CONFIG)
    add_app_src_to_path()


def load_corpus_records(path: Path) -> list[dict]:
    """Load the paragraph corpus exactly as the Demo's ``/initialize_elastic_from_json`` does.

    Args:
        path (Path): Path to the paragraph corpus JSON.

    Returns:
        list[dict]: One record per paragraph (``doc_id``, ``title``, ``content``, ``link``).
    """
    add_app_src_to_path()
    from utils import convert_kolzchut_paragraphs_corpus_to_json

    return convert_kolzchut_paragraphs_corpus_to_json(str(path))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

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
