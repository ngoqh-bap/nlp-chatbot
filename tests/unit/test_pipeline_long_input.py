"""Long-input behavior: bounded intent text, full raw text for entities."""

import importlib

import pytest


@pytest.mark.unit
def test_entities_use_full_raw_text_when_intent_input_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NLU_INTENT_MAX_CHARS", "30")
    import config
    import nlu.pipeline as pl

    importlib.reload(config)
    importlib.reload(pl)

    long = "B" * 200
    p = pl.NLPPipeline()
    captured: dict = {}

    real_extract = p.extract_entities

    def spy_extract(t: str):
        captured["raw_len"] = len(t)
        return real_extract(t)

    monkeypatch.setattr(p, "extract_entities", spy_extract)

    if p._intent_engine is None:
        pytest.skip("no intent engine")

    real_engine = p._intent_engine

    class _SpyIntent:
        def detect(self, intent_in, sm, norm):
            captured["intent_len"] = len(intent_in)
            return real_engine.detect(intent_in, sm, norm)

    monkeypatch.setattr(p, "_intent_engine", _SpyIntent())

    p.analyze_with_context(long, {})
    assert captured["raw_len"] == 200
    assert captured["intent_len"] <= 30


@pytest.mark.unit
def test_fallback_intent_returns_engine_top1_score(monkeypatch: pytest.MonkeyPatch) -> None:
    import nlu.pipeline as pl

    p = pl.NLPPipeline()
    if p._intent_engine is None:
        pytest.skip("no intent engine")

    class _FixedIntent:
        def detect(self, *_a, **_k):
            return "fallback", 0.41

    monkeypatch.setattr(p, "_intent_engine", _FixedIntent())
    out = p.analyze_with_context("hi", {})
    assert out["intent"] == "fallback" and out["score"] == 0.41


@pytest.mark.unit
def test_entity_engine_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NLU_ENTITY_ENGINE", "off")
    import config
    import nlu.pipeline as pl

    importlib.reload(config)
    importlib.reload(pl)

    p = pl.NLPPipeline()
    assert p.extract_entities("điểm chuẩn ngành kiến trúc") == []


@pytest.mark.unit
def test_merge_dedupes_identical_spans() -> None:
    from nlu.pipeline import NLPPipeline

    d = [
        {
            "label": "X",
            "text": "a",
            "start": 0,
            "end": 1,
            "source": "pattern",
        }
    ]
    n = [
        {
            "label": "X",
            "text": "a",
            "start": 0,
            "end": 1,
            "source": "model",
        }
    ]
    merged = NLPPipeline._merge_entity_lists(d, n)
    assert len(merged) == 1
    assert merged[0]["source"] == "pattern"
