"""Retrieval extensions to the ``webiks-hebrew-ragbot`` engine.

Title-aware retrieval re-scores the documents the engine's dense search finds with the
similarity between the query and each document's title. It is off unless
``IS_TITLE_AWARE_RETRIEVAL`` is set; see ``settings.py``.
"""
