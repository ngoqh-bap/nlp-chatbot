"""Tests for transformer NER helpers and engine (mocked HF)."""

import importlib

import pytest

from nlu.engines.ner_transformer import TransformerNerEngine, offsets_to_spans


@pytest.mark.unit
def test_offsets_to_spans_merges_bio() -> None:
    text = "ngành kiến trúc"
    offsets = [(0, 5), (6, 10), (11, 15)]
    tags = ["O", "B-TEN_NGANH", "I-TEN_NGANH"]
    spans = offsets_to_spans(text, offsets, tags)
    assert spans == [
        {
            "label": "TEN_NGANH",
            "text": "kiến trúc",
            "start": 6,
            "end": 15,
            "source": "model",
        }
    ]


@pytest.mark.unit
def test_offsets_to_spans_unicode_cafe() -> None:
    text = "tại café"
    # char indices: space at 3, café starts 4 ends 8 (4 code points)
    offsets = [(0, 3), (4, 8)]
    tags = ["O", "B-LOC"]
    spans = offsets_to_spans(text, offsets, tags)
    assert len(spans) == 1
    assert spans[0]["text"] == "café"
    assert spans[0]["start"] == 4
    assert spans[0]["end"] == 8


@pytest.mark.unit
def test_ner_skips_when_over_nlu_ner_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NLU_NER_MAX_CHARS", "10")
    import config

    importlib.reload(config)

    eng = object.__new__(TransformerNerEngine)
    assert eng.extract("x" * 50) == []  # type: ignore[attr-defined]


@pytest.mark.unit
def test_transformer_ner_merge_forward_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("NLU_DEVICE", "cpu")
    monkeypatch.delenv("NLU_NER_MAX_CHARS", raising=False)
    import config
    import torch
    import transformers

    importlib.reload(config)

    class FakeModel:
        config = type(
            "C",
            (),
            {
                "num_labels": 3,
                "id2label": {0: "O", 1: "B-PER", 2: "I-PER"},
                "max_position_embeddings": 512,
            },
        )()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            class Out:
                logits = torch.tensor(
                    [
                        [
                            [2.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                        ]
                    ]
                )

            return Out()

    class FakeTok:
        def __call__(self, text, **kwargs):
            n = len(text)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "offset_mapping": torch.tensor([[[0, 1], [1, n]]]),
            }

    def _fake_tok(*_a, **_k):
        return FakeTok()

    def _fake_model(*_a, **_k):
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake_tok)
    monkeypatch.setattr(
        transformers.AutoModelForTokenClassification,
        "from_pretrained",
        _fake_model,
    )

    eng = TransformerNerEngine(str(tmp_path))
    out = eng.extract("ab")
    assert isinstance(out, list)
