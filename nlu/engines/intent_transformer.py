"""Transformer intent engine pieces and PhoBERT runtime."""

from __future__ import annotations

from typing import Any, Dict

from config import (
    get_intent_budget_ms,
    get_intent_margin_M,
    get_intent_threshold_T,
    get_nlu_device,
    get_nlu_intent_max_chars,
    get_nlu_use_autocast,
)

from .base import NormalizeFn
from .utils import run_with_budget_ms


def decide_intent(labels: list[str], probs: list[float], t: float, m: float) -> tuple[str, float]:
    if not labels or not probs:
        return "fallback", 0.0
    ranked = sorted(zip(probs, labels), key=lambda x: -x[0])
    top1_p, top1_l = ranked[0]
    top2_p = ranked[1][0] if len(ranked) > 1 else 0.0
    if top1_p < t or (top1_p - top2_p) < m:
        return "fallback", top1_p
    return top1_l, top1_p


class TransformerIntentEngine:
    """HF sequence-classification intent (e.g. fine-tuned PhoBERT)."""

    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device_s = get_nlu_device()
        if device_s == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model.to(self._device)
        else:
            self._device = torch.device("cpu")
            self._model.to(self._device)
        self._model.eval()
        self._use_autocast = bool(get_nlu_use_autocast() and self._device.type == "cuda")

    def detect(
        self, text: str, synonym_map: Dict[str, str], normalize_for_kw_fn: NormalizeFn
    ) -> tuple[str, float]:
        _ = (synonym_map, normalize_for_kw_fn)
        max_c = get_nlu_intent_max_chars()
        intent_text = text if max_c <= 0 else text[:max_c]
        t = get_intent_threshold_T()
        m = get_intent_margin_M()

        def run() -> tuple[str, float]:
            return self._infer(intent_text, t, m)

        return run_with_budget_ms(
            get_intent_budget_ms(),
            run,
            on_timeout=("fallback", 0.0),
        )

    def _infer(self, text: str, t: float, m: float) -> tuple[str, float]:
        import torch
        import torch.nn.functional as F

        try:
            if not (text or "").strip():
                return "fallback", 0.0
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            cfg = self._model.config
            labels = [str(cfg.id2label[i]) for i in range(cfg.num_labels)]
            with torch.no_grad():
                if self._use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self._model(**enc).logits
                else:
                    logits = self._model(**enc).logits
            probs = F.softmax(logits[0], dim=-1).detach().cpu().tolist()
            return decide_intent(labels, probs, t, m)
        except Exception:
            return "fallback", 0.0
