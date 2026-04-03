"""Tests for NER JSONL label maps (used by train_ner_phobert.py)."""

import pytest

from nlu.datasets.ner_jsonl import build_label_maps, build_label_maps_from_rows


@pytest.mark.unit
def test_ner_label_maps_o_first(tmp_path) -> None:
    p = tmp_path / "sample.jsonl"
    p.write_text(
        '{"tokens":["a","b"],"ner_tags":["B-X","O"]}\n'
        '{"tokens":["c"],"ner_tags":["O"]}\n',
        encoding="utf-8",
    )
    label2id, id2label = build_label_maps(p)
    assert id2label[0] == "O"
    assert label2id["O"] == 0
    assert label2id["B-X"] == 1


@pytest.mark.unit
def test_build_label_maps_from_rows_o_first() -> None:
    rows = [
        (["a", "b"], ["B-X", "O"]),
        (["c"], ["O"]),
    ]
    label2id, id2label = build_label_maps_from_rows(rows)
    assert id2label[0] == "O"


@pytest.mark.unit
def test_build_label_maps_requires_o(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"tokens":["a"],"ner_tags":["B-X"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tag 'O'"):
        build_label_maps(p)
