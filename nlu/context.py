"""Bounded, deterministic context shaping for intent (multi-turn)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

_MAX_DROP_SNIPPET = 500


def _user_line(message: str) -> str:
    return f"User: {message}"


class ContextProcessor:
    def __init__(self, max_chars: int, turns_for_model: int) -> None:
        self.max_chars = max(1, int(max_chars))
        self.turns_for_model = max(1, int(turns_for_model))

    def build_intent_input(self, message: str, current_context: Dict[str, Any]) -> str:
        hist: List[Dict[str, Any]] = list(current_context.get("conversation_history") or [])
        summary = str(current_context.get("context_summary") or "")
        if not summary.strip() and not hist:
            return message
        window = hist[-self.turns_for_model :]
        window_text = "\n".join(
            str(e.get("message", "") or "") for e in window
        ).strip()
        parts: List[str] = []
        if summary.strip():
            parts.append(summary.strip())
        if window_text:
            parts.append(window_text)
        parts.append(_user_line(message))
        full = "\n".join(parts)
        if len(full) > self.max_chars:
            return full[-self.max_chars :]
        return full

    def build_intent_input_and_update_context(
        self, message: str, current_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        ctx = deepcopy(current_context)
        working_hist: List[Dict[str, Any]] = list(ctx.get("conversation_history") or [])
        working_summary = str(ctx.get("context_summary") or "")
        if not working_summary.strip() and not working_hist:
            m = message[: self.max_chars] if len(message) > self.max_chars else message
            return m, ctx

        while True:
            window = working_hist[-self.turns_for_model :]
            window_text = "\n".join(
                str(e.get("message", "") or "") for e in window
            ).strip()
            parts: List[str] = []
            if working_summary.strip():
                parts.append(working_summary.strip())
            if window_text:
                parts.append(window_text)
            parts.append(_user_line(message))
            full = "\n".join(parts)

            if len(full) <= self.max_chars:
                ctx["conversation_history"] = working_hist
                ctx["context_summary"] = working_summary
                return full, ctx

            if working_hist:
                dropped = working_hist.pop(0)
                snippet = str(dropped.get("message", "") or "")[:_MAX_DROP_SNIPPET]
                working_summary = (
                    (working_summary + "\n" + snippet).strip()
                    if working_summary
                    else snippet
                )
                continue

            if working_summary.strip():
                need = self.max_chars - len(_user_line(message)) - 1
                if need <= 0:
                    u = _user_line(message)
                    ctx["conversation_history"] = working_hist
                    ctx["context_summary"] = working_summary
                    return u[: self.max_chars], ctx
                if len(working_summary) > need:
                    working_summary = working_summary[-need:]
                continue

            u = _user_line(message)
            ctx["conversation_history"] = working_hist
            ctx["context_summary"] = working_summary
            return u[: self.max_chars], ctx
