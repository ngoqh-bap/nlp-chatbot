import pytest


@pytest.mark.unit
def test_decide_intent_low_top1():
    from nlu.engines.intent_transformer import decide_intent

    # Top-1 prob is 0.8 ("b"); below threshold 0.9 → fallback with top-1 score (IntentDetector parity).
    intent, score = decide_intent(["a", "b"], [0.2, 0.8], t=0.9, m=0.0)
    assert intent == "fallback" and score == 0.8


@pytest.mark.unit
def test_decide_intent_small_margin():
    from nlu.engines.intent_transformer import decide_intent

    intent, score = decide_intent(["a", "b"], [0.51, 0.49], t=0.0, m=0.05)
    assert intent == "fallback" and score == 0.51
