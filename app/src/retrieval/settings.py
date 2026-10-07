"""Retrieval settings.

- ``IS_TITLE_AWARE_RETRIEVAL`` (environment): ``true`` (any case) switches title-aware
  retrieval on. Unset or any other value keeps the engine's own search.
- ``TITLE_WEIGHT`` (code): weight of the query-title similarity in title-aware retrieval.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

FLAG_VARIABLE = "IS_TITLE_AWARE_RETRIEVAL"

# Selected on the dev questions by MRR@10 (evaluation/reports/weight_sweep_dev.json); the test
# comparison in evaluation/reports/final_test.json measured this weight.
TITLE_WEIGHT = 0.85


@dataclass(frozen=True)
class RetrievalSettings:
    """How documents are retrieved.

    Attributes:
        title_aware (bool): Whether title-aware retrieval is on.
        title_weight (float | None): Weight of the query-title similarity; None when off.
    """

    title_aware: bool
    title_weight: float | None = None


def load_settings(environ: Mapping[str, str] = os.environ) -> RetrievalSettings:
    """Read the retrieval settings.

    Args:
        environ (Mapping[str, str]): Environment variables.

    Returns:
        RetrievalSettings: The settings.
    """
    if environ.get(FLAG_VARIABLE, "false").strip().lower() != "true":
        return RetrievalSettings(title_aware=False)
    return RetrievalSettings(title_aware=True, title_weight=TITLE_WEIGHT)
