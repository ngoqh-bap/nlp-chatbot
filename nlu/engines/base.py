"""Protocols for pluggable NLU intent and NER engines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, Tuple, runtime_checkable

IntentResult = Tuple[str, float]

NormalizeFn = Callable[..., str]


@runtime_checkable
class IntentEngine(Protocol):
    def detect(self, text: str, synonym_map: Dict[str, str], normalize_for_kw_fn: NormalizeFn) -> IntentResult:
        ...


@runtime_checkable
class NerEngine(Protocol):
    def extract(self, text: str) -> List[Dict[str, Any]]:
        ...
