import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

APP_SRC = Path(__file__).resolve().parents[1] / "app" / "src"
REPORTS = Path(__file__).resolve().parents[1] / "evaluation" / "reports"
sys.path.insert(0, str(APP_SRC))

# The engine reads its document definitions when it is first imported.
with patch.dict(os.environ, {"DOCUMENT_DEFINITION_CONFIG": str(APP_SRC / "doc-config.json")}):
    from retrieval import factory
    from retrieval.settings import TITLE_WEIGHT, RetrievalSettings, load_settings


def test_defaults_to_the_engine_search():
    assert load_settings({}) == RetrievalSettings(title_aware=False, title_weight=None)


@pytest.mark.parametrize("flag", ["true", "TRUE", " True "])
def test_flag_on(flag):
    settings = load_settings({"IS_TITLE_AWARE_RETRIEVAL": flag})

    assert settings == RetrievalSettings(title_aware=True, title_weight=TITLE_WEIGHT)


@pytest.mark.parametrize(
    ("report", "keys"),
    [
        ("weight_sweep_dev.json", ("selection", "weight")),
        ("final_test.json", ("setup", "title_weight")),
    ],
)
def test_title_weight_is_the_evaluated_weight(report, keys):
    value = json.loads((REPORTS / report).read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]

    assert TITLE_WEIGHT == value


@pytest.mark.parametrize("flag", ["false", "", "1", "yes"])
def test_flag_off(flag):
    assert load_settings({"IS_TITLE_AWARE_RETRIEVAL": flag}).title_aware is False


@pytest.mark.parametrize("flag", ["true", "false"])
def test_weight_variable_is_ignored(flag):
    settings = load_settings({"IS_TITLE_AWARE_RETRIEVAL": flag, "TITLE_FUSION_WEIGHT": "abc"})

    assert settings.title_weight == (TITLE_WEIGHT if flag == "true" else None)


def test_reads_the_process_environment_by_default():
    with patch.dict(os.environ, {"IS_TITLE_AWARE_RETRIEVAL": "true"}):
        assert load_settings() == RetrievalSettings(title_aware=True, title_weight=TITLE_WEIGHT)


@pytest.fixture
def no_engine(monkeypatch):
    monkeypatch.setattr(factory, "engine", None)


def test_factory_off_uses_the_engine_factory(no_engine, caplog):
    with (
        patch.object(factory, "base_engine_factory", return_value="base") as base,
        patch.object(factory, "TitleAwareEngine") as title_aware,
        caplog.at_level(logging.INFO, logger="retrieval.factory"),
    ):
        engine = factory.engine_factory("llm", "es", RetrievalSettings(title_aware=False))

    assert engine == "base"
    base.assert_called_once_with("llm", "es")
    title_aware.assert_not_called()
    assert "dense" in caplog.text


def test_factory_on_builds_one_title_aware_engine(no_engine, caplog):
    settings = RetrievalSettings(title_aware=True, title_weight=0.35)
    with (
        patch.object(factory, "base_engine_factory") as base,
        patch.object(factory, "TitleAwareEngine") as title_aware,
        caplog.at_level(logging.INFO, logger="retrieval.factory"),
    ):
        first = factory.engine_factory("llm", "es", settings)
        second = factory.engine_factory("llm", "es", settings)

    assert first is second is title_aware.return_value
    title_aware.assert_called_once_with(llms_client="llm", es_client="es", title_weight=0.35)
    base.assert_not_called()
    assert "title-aware, title weight 0.35" in caplog.text


def test_factory_reads_the_environment_when_no_settings_are_given(no_engine):
    with (
        patch.dict(os.environ, {"IS_TITLE_AWARE_RETRIEVAL": "false"}),
        patch.object(factory, "base_engine_factory", return_value="base"),
    ):
        assert factory.engine_factory("llm", "es") == "base"


def test_factory_on_from_the_environment_uses_the_selected_weight(no_engine):
    with (
        patch.dict(os.environ, {"IS_TITLE_AWARE_RETRIEVAL": "true"}),
        patch.object(factory, "TitleAwareEngine") as title_aware,
    ):
        factory.engine_factory("llm", "es")

    title_aware.assert_called_once_with(
        llms_client="llm", es_client="es", title_weight=TITLE_WEIGHT
    )
