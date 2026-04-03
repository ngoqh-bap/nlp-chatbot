"""Tests for ContextProcessor."""

from nlu.context import ContextProcessor


def test_model_input_is_bounded_by_max_chars():
    cp = ContextProcessor(max_chars=50, turns_for_model=3)
    ctx = {"conversation_history": [{"message": "A" * 100}], "context_summary": "S" * 100}
    out = cp.build_intent_input("hi", ctx)
    assert len(out) <= 50


def test_history_retention_is_not_modified_by_processor():
    cp = ContextProcessor(max_chars=200, turns_for_model=3)
    hist = [{"message": "m1"}, {"message": "m2"}]
    ctx = {"conversation_history": hist.copy()}
    _ = cp.build_intent_input("m3", ctx)
    assert ctx["conversation_history"] == hist


def test_overflow_updates_summary_deterministically():
    cp = ContextProcessor(max_chars=40, turns_for_model=10)
    ctx = {"conversation_history": [{"message": "m1 " * 50}, {"message": "m2"}], "context_summary": ""}
    out1, ctx1 = cp.build_intent_input_and_update_context("hi", ctx)
    out2, ctx2 = cp.build_intent_input_and_update_context("hi", ctx)
    assert out1 == out2
    assert ctx1["context_summary"] == ctx2["context_summary"]
    assert len(out1) <= 40
