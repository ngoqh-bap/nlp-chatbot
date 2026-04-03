import importlib

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


@pytest.mark.unit
def test_transformer_intent_engine_forward_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("NLU_DEVICE", "cpu")
    import torch

    import config

    importlib.reload(config)

    class FakeModel:
        config = type(
            "C",
            (),
            {
                "num_labels": 2,
                "id2label": {0: "intent_a", 1: "intent_b"},
            },
        )()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            class Out:
                logits = torch.tensor([[0.0, 3.0]])

            return Out()

    class FakeTok:
        def __call__(self, _text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    import transformers

    def _fake_tok(*_a, **_k):
        return FakeTok()

    def _fake_model(*_a, **_k):
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake_tok)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        _fake_model,
    )

    import nlu.engines.intent_transformer as it_mod

    eng = it_mod.TransformerIntentEngine(str(tmp_path))
    intent, score = eng.detect("xin chào", {}, lambda s: s)
    assert intent == "intent_b"
    assert 0.0 < score <= 1.0
