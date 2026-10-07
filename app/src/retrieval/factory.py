"""Create the search engine the settings ask for."""

import logging

from elasticsearch import Elasticsearch
from webiks_hebrew_ragbot.engine import Engine
from webiks_hebrew_ragbot.engine import engine_factory as base_engine_factory
from webiks_hebrew_ragbot.llm_client import LLMClient

from .settings import RetrievalSettings, load_settings
from .title_aware_engine import TitleAwareEngine

logger = logging.getLogger(__name__)

engine = None


def engine_factory(
    llms_client: LLMClient,
    es_client: Elasticsearch | None = None,
    settings: RetrievalSettings | None = None,
) -> Engine:
    """Return the engine, created on the first call.

    With title-aware retrieval off, this is the engine package's own ``engine_factory``.

    Args:
        llms_client (LLMClient): The LLM client.
        es_client (Elasticsearch | None): The Elasticsearch client.
        settings (RetrievalSettings | None): Retrieval settings; read from the
            environment when None.

    Returns:
        Engine: The engine; a ``TitleAwareEngine`` when title-aware retrieval is on.
    """
    global engine
    if settings is None:
        settings = load_settings()
    if not settings.title_aware:
        logger.info("Retrieval: dense (engine default)")
        return base_engine_factory(llms_client, es_client)
    if engine is None:
        logger.info("Retrieval: title-aware, title weight %s", settings.title_weight)
        engine = TitleAwareEngine(
            llms_client=llms_client, es_client=es_client, title_weight=settings.title_weight
        )
    return engine
