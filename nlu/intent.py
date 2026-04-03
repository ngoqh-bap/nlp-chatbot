import math
from typing import Dict, List, Tuple

from .engines.base import NormalizeFn
from .preprocess import tokenize_and_map


class LegacyTfidfIntentEngine:
    """Adapter exposing `IntentDetector` through the `IntentEngine` protocol."""

    __slots__ = ("_detector",)

    def __init__(self, detector: "IntentDetector") -> None:
        self._detector = detector

    def detect(
        self, text: str, synonym_map: Dict[str, str], normalize_for_kw_fn: NormalizeFn
    ) -> Tuple[str, float]:
        return self._detector.detect(text, synonym_map, normalize_for_kw_fn)

DEFAULT_INTENT_THRESHOLD = 0.3
DEFAULT_INTENT_MARGIN = 0.0
# Many intent labels: low temperature sharpens softmax so top-1 probability stays meaningful.
LEGACY_SOFTMAX_TEMPERATURE = 0.12


def _softmax(values: List[float], temperature: float) -> List[float]:
    if not values:
        return []
    t = temperature if temperature > 1e-9 else 1e-9
    scaled = [v / t for v in values]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _compute_idf(samples: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    n_docs = len(samples)

    for toks in samples:
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, c in df.items():
        idf[t] = math.log((1 + n_docs) / (1 + c)) + 1.0
    return idf


def _tf(toks: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1

    total = float(len(toks)) or 1.0
    return {t: c / total for t, c in counts.items()}


def _centroid(vecs: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for v in vecs:
        for k, val in v.items():
            agg[k] = agg.get(k, 0.0) + val

    norm = math.sqrt(sum(v * v for v in agg.values())) or 1.0
    return {k: v / norm for k, v in agg.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a

    s = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            s += va * vb
    return s


class IntentDetector:
    def __init__(
        self,
        intent_samples: Dict[str, List[List[str]]],
        threshold: float = DEFAULT_INTENT_THRESHOLD,
        margin_m: float = DEFAULT_INTENT_MARGIN,
    ) -> None:
        self.intent_samples = intent_samples
        self.threshold = threshold
        self.margin_m = margin_m

        self.idf: Dict[str, float] = {}
        self.intent_centroids: Dict[str, Dict[str, float]] = {}
        self._intent_order: List[str] = []
        self._build_intent_centroids()

    def _tfidf_vec(self, toks: List[str]) -> Dict[str, float]:
        tf = _tf(toks)
        vec = {t: tf[t] * self.idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _build_intent_centroids(self) -> None:
        all_samples: List[List[str]] = []
        for samples in self.intent_samples.values():
            all_samples.extend(samples)

        self.idf = _compute_idf(all_samples) if all_samples else {}

        centroids: Dict[str, Dict[str, float]] = {}
        for intent, samples in self.intent_samples.items():
            vecs = [self._tfidf_vec(s) for s in samples]
            centroids[intent] = _centroid(vecs) if vecs else {}

        self.intent_centroids = centroids
        self._intent_order = list(self.intent_centroids.keys())

    def detect(
        self, text: str, synonym_map: Dict[str, str], normalize_for_kw_fn: NormalizeFn
    ) -> Tuple[str, float]:
        if not (text or "").strip():
            return "fallback", 0.0

        q_tokens = tokenize_and_map(text, synonym_map)
        if not q_tokens:
            return "fallback", 0.0

        q_vec = self._tfidf_vec(q_tokens)

        if not self._intent_order:
            return "fallback", 0.0

        similarities = [
            _cosine(q_vec, self.intent_centroids[i]) for i in self._intent_order
        ]
        probs = _softmax(similarities, LEGACY_SOFTMAX_TEMPERATURE)
        ranked = sorted(
            zip(self._intent_order, probs),
            key=lambda x: -x[1],
        )
        top1_label, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0

        if top1 < self.threshold or (top1 - top2) < self.margin_m:
            return "fallback", top1
        return top1_label, top1
