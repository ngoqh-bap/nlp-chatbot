"""NLU-related config defaults."""

from config import (
    get_entity_budget_ms,
    get_intent_budget_ms,
    get_intent_model_path,
    get_nlu_entity_engine,
    get_nlu_intent_engine,
    get_nlu_device,
    get_nlu_intent_max_chars,
    get_nlu_use_autocast,
    get_ner_model_path,
)


def test_default_intent_engine_is_legacy():
    assert get_nlu_intent_engine() == "legacy"


def test_default_entity_engine_is_deterministic():
    assert get_nlu_entity_engine() == "deterministic"


def test_intent_model_path_default_empty(monkeypatch):
    monkeypatch.delenv("NLU_INTENT_MODEL_DIR", raising=False)
    monkeypatch.delenv("INTENT_MODEL_PATH", raising=False)
    assert get_intent_model_path() == ""


def test_ner_model_path_default_empty(monkeypatch):
    monkeypatch.delenv("NLU_NER_MODEL_DIR", raising=False)
    monkeypatch.delenv("NER_MODEL_PATH", raising=False)
    assert get_ner_model_path() == ""


def test_intent_model_path_alias(monkeypatch):
    monkeypatch.delenv("NLU_INTENT_MODEL_DIR", raising=False)
    monkeypatch.setenv("INTENT_MODEL_PATH", "C:/models/intent")
    assert get_intent_model_path() == "C:/models/intent"


def test_ner_model_path_alias(monkeypatch):
    monkeypatch.delenv("NLU_NER_MODEL_DIR", raising=False)
    monkeypatch.setenv("NER_MODEL_PATH", "C:/models/ner")
    assert get_ner_model_path() == "C:/models/ner"


def test_nlu_intent_max_chars_default_disabled(monkeypatch):
    monkeypatch.delenv("NLU_INTENT_MAX_CHARS", raising=False)
    assert get_nlu_intent_max_chars() == 0


def test_nlu_device_valid(monkeypatch):
    monkeypatch.delenv("NLU_DEVICE", raising=False)
    assert get_nlu_device() in ("cpu", "cuda")


def test_nlu_device_env_overrides(monkeypatch):
    monkeypatch.setenv("NLU_DEVICE", "CPU")
    assert get_nlu_device() == "cpu"


def test_nlu_autocast_env_overrides(monkeypatch):
    monkeypatch.setenv("NLU_AUTOCAST", "false")
    assert get_nlu_use_autocast() is False


def test_intent_budget_ms_prefers_nlu_env(monkeypatch):
    monkeypatch.setenv("NLU_INTENT_BUDGET_MS", "111")
    monkeypatch.setenv("INTENT_INFERENCE_BUDGET_MS", "222")
    assert get_intent_budget_ms() == 111


def test_entity_budget_ms_prefers_nlu_env(monkeypatch):
    monkeypatch.setenv("NLU_ENTITY_BUDGET_MS", "333")
    monkeypatch.setenv("ENTITY_INFERENCE_BUDGET_MS", "444")
    assert get_entity_budget_ms() == 333
