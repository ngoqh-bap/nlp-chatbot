"""Transformer token-classification NER (e.g. PhoBERT BIO)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

_logger = logging.getLogger(__name__)

from config import (
    get_entity_budget_ms,
    get_ner_model_path,
    get_nlu_device,
    get_nlu_ner_max_chars,
    get_nlu_use_autocast,
)

from .utils import run_with_budget_ms


def offsets_to_spans(
    text: str,
    offsets: Sequence[Tuple[int, int]],
    tags: Sequence[str],
) -> List[Dict[str, Any]]:
    """Merge BIO tags using HF char offsets (start inclusive, end exclusive)."""
    entities: List[Dict[str, Any]] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = (tags[i] or "O").strip()
        if tag == "O" or not tag.startswith("B-"):
            i += 1
            continue
        lab = tag[2:]
        if i >= len(offsets):
            break
        start_char, end_char = offsets[i][0], offsets[i][1]
        j = i + 1
        while j < n:
            t2 = (tags[j] or "O").strip()
            if t2 == f"I-{lab}":
                if j < len(offsets):
                    end_char = offsets[j][1]
                j += 1
            else:
                break
        surface = text[start_char:end_char]
        entities.append(
            {
                "label": lab,
                "text": surface,
                "start": start_char,
                "end": end_char,
                "source": "model",
            }
        )
        i = j
    return entities


def _max_token_length_for_model(tokenizer: Any, model: Any) -> int:
    tmax = getattr(tokenizer, "model_max_length", 10_000_000)
    if not isinstance(tmax, int) or tmax > 1_000_000:
        tmax = 10_000_000
    pmax = getattr(model.config, "max_position_embeddings", tmax)
    return min(tmax, pmax)


class TransformerNerEngine:
    """HF token classification on raw user `text` only."""

    def __init__(self, model_dir: str | None = None) -> None:
        path = (model_dir or get_ner_model_path()).strip()
        if not path:
            raise ValueError("NER model path is required")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self._model = AutoModelForTokenClassification.from_pretrained(path)
        device_s = get_nlu_device()
        if device_s == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model.to(self._device)
        else:
            self._device = torch.device("cpu")
            self._model.to(self._device)
        self._model.eval()
        self._use_autocast = bool(get_nlu_use_autocast() and self._device.type == "cuda")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        cap = get_nlu_ner_max_chars()
        if cap > 0 and len(text) > cap:
            return []

        def run() -> List[Dict[str, Any]]:
            return self._forward(text)

        return run_with_budget_ms(get_entity_budget_ms(), run, on_timeout=[])

    def _forward(self, text: str) -> List[Dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        if not (text or "").strip():
            return []
        try:
            max_len = _max_token_length_for_model(self._tokenizer, self._model)
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
                padding=False,
            )
            offs = enc.pop("offset_mapping")[0].tolist()
            if not enc["input_ids"].numel():
                return []
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                if self._use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self._model(**enc).logits
                else:
                    logits = self._model(**enc).logits
            pred = logits[0].argmax(dim=-1).tolist()
            idmap = self._model.config.id2label
            tags: List[str] = []
            for pid in pred:
                lab = idmap.get(pid)
                if lab is None:
                    lab = idmap.get(str(pid), "O")
                tags.append(str(lab))
            char_offsets = [(int(a), int(b)) for a, b in offs]
            max_char = 0
            for a, b in char_offsets:
                if b > max_char:
                    max_char = b
            processed_len = max_char
            ents = offsets_to_spans(text, char_offsets, tags)
            return [e for e in ents if e["end"] <= processed_len]
        except Exception:
            _logger.warning("nlu_ner_forward_failed", exc_info=True)
            return []
