import csv
import os
from typing import List, Dict, Tuple, Any, Optional

try:
    from .preprocess import normalize_text as ext_normalize_text
    from .preprocess import tokenize_and_map as ext_tokenize_and_map
except ImportError:
    ext_normalize_text = None
    ext_tokenize_and_map = None

try:
    from .intent import IntentDetector, LegacyTfidfIntentEngine
except ImportError:
    IntentDetector = None
    LegacyTfidfIntentEngine = None

try:
    from .context import ContextProcessor
except ImportError:
    ContextProcessor = None

try:
    from .entities import EntityExtractor
except ImportError:
    EntityExtractor = None

from config import (
    DATA_DIR,
    get_context_max_chars,
    get_context_turns_for_model,
    get_intent_margin_M,
    get_intent_model_path,
    get_intent_threshold,
    get_nlu_intent_engine,
)

from .engines.base import IntentEngine, PlaceholderTransformerIntentEngine

DEFAULT_INTENT_THRESHOLD = get_intent_threshold()


def _normalize_text(text) -> str:
    if ext_normalize_text is not None:
        return ext_normalize_text(text)
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text.lower().strip()


def _load_synonyms(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.isfile(path):
        return mapping

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or (row[0] and row[0].strip().startswith('#')) or len(row) < 3:
                continue
            entity, canonical, alias = row[0].strip(), row[1].strip(), row[2].strip()
            if entity == "entity" or not alias or not canonical:
                continue
            alias_norm, canonical_norm = _normalize_text(alias), _normalize_text(canonical)
            if alias_norm and canonical_norm:
                mapping[alias_norm] = canonical_norm
    return mapping


class NLPPipeline:
    def __init__(self, data_dir: str = DATA_DIR, intent_threshold: float = DEFAULT_INTENT_THRESHOLD) -> None:
        self.data_dir = data_dir
        self.intent_threshold = intent_threshold
        self.syn_map = _load_synonyms(os.path.join(data_dir, "synonym.csv"))
        self.intent_samples = self._load_intent_samples(os.path.join(data_dir, "intent.csv"))


        self._intent_detector: Optional[IntentDetector] = (
            IntentDetector(
                self.intent_samples,
                self.intent_threshold,
                margin_m=get_intent_margin_M(),
            )
            if IntentDetector is not None else None
        )
        self._intent_engine_mode = get_nlu_intent_engine()
        self._intent_engine: Optional[IntentEngine] = None
        if self._intent_engine_mode == "transformer":
            model_path = get_intent_model_path()
            if model_path and os.path.isdir(model_path):
                try:
                    from .engines.intent_transformer import TransformerIntentEngine

                    self._intent_engine = TransformerIntentEngine(model_path)
                except Exception:
                    self._intent_engine = PlaceholderTransformerIntentEngine()
            else:
                self._intent_engine = PlaceholderTransformerIntentEngine()
        elif (
            self._intent_detector is not None
            and LegacyTfidfIntentEngine is not None
        ):
            self._intent_engine = LegacyTfidfIntentEngine(self._intent_detector)
        self._entity_extractor: Optional[EntityExtractor] = (
            EntityExtractor(self.data_dir, os.path.join(data_dir, "entity.json"), self.syn_map)
            if EntityExtractor is not None else None
        )
        self._context_processor: Optional[ContextProcessor] = (
            ContextProcessor(
                max_chars=get_context_max_chars(),
                turns_for_model=get_context_turns_for_model(),
            )
            if ContextProcessor is not None else None
        )

    def _load_intent_samples(self, path: str) -> Dict[str, List[List[str]]]:
        intent_to_samples: Dict[str, List[List[str]]] = {}
        if not os.path.isfile(path):
            return intent_to_samples

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                utt = _normalize_text(r.get("utterance") or "")
                intent = (r.get("intent") or "").strip()
                if not utt or not intent:
                    continue
                toks = ext_tokenize_and_map(utt, self.syn_map) if ext_tokenize_and_map else utt.split()
                intent_to_samples.setdefault(intent, []).append(toks)
        return intent_to_samples

    def detect_intent(self, text: str) -> Tuple[str, float]:
        if self._intent_engine is None:
            return "fallback", 0.0
        return self._intent_engine.detect(text, self.syn_map, _normalize_text)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self._entity_extractor is None:
            return []
        return self._entity_extractor.extract(text)

    def analyze(self, text: str) -> Dict[str, Any]:
        return self.analyze_with_context(text, {})

    def analyze_with_context(
        self, text: str, current_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = current_context if current_context is not None else {}
        if self._context_processor is not None:
            intent_input = self._context_processor.build_intent_input(text, ctx)
        else:
            intent_input = text
        intent, score = (
            self._intent_engine.detect(intent_input, self.syn_map, _normalize_text)
            if self._intent_engine is not None
            else ("fallback", 0.0)
        )
        entities = self.extract_entities(text)
        return {"intent": intent, "score": score, "entities": entities}
