"""Pipeline edge cases aligned with specs."""

from nlu.pipeline import NLPPipeline


def test_empty_input_returns_fallback_and_no_entities():
    p = NLPPipeline()
    out = p.analyze("")
    assert out["intent"] == "fallback"
    assert out["score"] == 0.0
    assert out["entities"] == []


def test_symbols_only_returns_fallback_and_no_entities():
    p = NLPPipeline()
    out = p.analyze("!!!")
    assert out["intent"] == "fallback"
    assert out["score"] == 0.0
    assert out["entities"] == []
