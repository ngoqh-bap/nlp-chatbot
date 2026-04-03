"""NLU-related config defaults."""

from config import (
    get_nlu_entity_engine,
    get_nlu_intent_engine,
)


def test_default_intent_engine_is_legacy():
    assert get_nlu_intent_engine() == "legacy"


def test_default_entity_engine_is_deterministic():
    assert get_nlu_entity_engine() == "deterministic"
