"""NLU pipeline intent engine selection."""

import importlib

import pytest


@pytest.mark.unit
def test_default_intent_engine_is_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without NLU_INTENT_ENGINE, pipeline should use the legacy TF-IDF intent adapter."""
    monkeypatch.delenv("NLU_INTENT_ENGINE", raising=False)

    import config
    import nlu.intent as nlu_intent
    import nlu.pipeline as nlu_pipeline

    importlib.reload(config)
    importlib.reload(nlu_intent)
    importlib.reload(nlu_pipeline)

    from nlu.intent import LegacyTfidfIntentEngine
    from nlu.pipeline import NLPPipeline

    p = NLPPipeline()
    assert isinstance(p._intent_engine, LegacyTfidfIntentEngine)
    assert p._intent_engine_mode == "legacy"
