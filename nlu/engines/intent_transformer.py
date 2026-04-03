"""Transformer intent engine pieces (full engine comes in later tasks)."""


def decide_intent(labels: list[str], probs: list[float], t: float, m: float) -> tuple[str, float]:
    if not labels or not probs:
        return "fallback", 0.0
    ranked = sorted(zip(probs, labels), key=lambda x: -x[0])
    top1_p, top1_l = ranked[0]
    top2_p = ranked[1][0] if len(ranked) > 1 else 0.0
    if top1_p < t or (top1_p - top2_p) < m:
        return "fallback", top1_p
    return top1_l, top1_p
