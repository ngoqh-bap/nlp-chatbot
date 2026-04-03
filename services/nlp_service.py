from typing import Dict, Any

from config import get_intent_threshold, get_context_history_limit
from nlu.pipeline import NLPPipeline


class ContextStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Dict[str, Any]:
        return self._store.get(session_id, {})

    def set(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self._store[session_id] = context
        return context

    def reset(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]

    def append_history(self, session_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self.get(session_id)
        hist = ctx.get("conversation_history", []) + [entry]

        limit = get_context_history_limit()
        if len(hist) > limit:
            hist = hist[-limit:]

        ctx["conversation_history"] = hist
        self.set(session_id, ctx)
        return ctx


class NLPService:
    def __init__(self) -> None:
        self.pipeline = NLPPipeline()
        self.context_store = ContextStore()
        self.intent_threshold = get_intent_threshold()

    def analyze_message(self, message: str) -> Dict[str, Any]:
        return self.pipeline.analyze(message)

    def handle_message(self, message: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        from services import csv_service as csvs

        analysis = self.pipeline.analyze_with_context(message, current_context)

        if analysis["intent"] == "fallback" or analysis["score"] < self.intent_threshold:
            response = csvs.handle_fallback_query(message, current_context)
        else:
            response = csvs.handle_intent_query(analysis, current_context, message)

        return {"analysis": analysis, "response": response}

    def get_context(self, session_id: str) -> Dict[str, Any]:
        return self.context_store.get(session_id)

    def set_context(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.context_store.set(session_id, context)

    def reset_context(self, session_id: str) -> None:
        self.context_store.reset(session_id)

    def append_history(self, session_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self.context_store.append_history(session_id, entry)


nlp_service = NLPService()


def get_nlp_service() -> NLPService:
    return nlp_service
