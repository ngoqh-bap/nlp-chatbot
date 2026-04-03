"""Load word-aligned BIO NER examples from JSON Lines for PhoBERT token classification."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_ner_jsonl(path: Path) -> Iterator[tuple[list[str], list[str]]]:
    """Yield ``(tokens, ner_tags)`` per non-empty line."""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj.get("tokens") or []
            tags = obj.get("ner_tags") or []
            if len(tokens) != len(tags):
                raise ValueError(f"{path}:{line_no}: len(tokens) != len(ner_tags)")
            if not tokens:
                continue
            yield tokens, tags


def load_ner_jsonl(path: Path) -> list[tuple[list[str], list[str]]]:
    """Load all examples; raises if file has no usable rows."""
    rows = list(iter_ner_jsonl(path))
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def _labels_to_maps(labels: set[str]) -> tuple[dict[str, int], dict[int, str]]:
    if "O" not in labels:
        raise ValueError("Corpus must include tag 'O' in ner_tags")
    rest = sorted(x for x in labels if x != "O")
    ordered = ["O"] + rest
    label2id = {lab: i for i, lab in enumerate(ordered)}
    id2label = dict(enumerate(ordered))
    return label2id, id2label


def build_label_maps_from_rows(
    rows: list[tuple[list[str], list[str]]],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build label maps from in-memory examples; **O** is always id 0."""
    if not rows:
        raise ValueError("No rows")
    labels: set[str] = set()
    for _tokens, tags in rows:
        labels.update(tags)
    return _labels_to_maps(labels)


def build_label_maps(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Collect BIO labels from JSONL file; **O** is always id 0."""
    labels: set[str] = set()
    n = 0
    for _tokens, tags in iter_ner_jsonl(path):
        n += 1
        labels.update(tags)
    if n == 0:
        raise ValueError(f"No examples in {path}")
    return _labels_to_maps(labels)
