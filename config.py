import os
from typing import List

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

INTENT_THRESHOLD_DEFAULT: float = 0.25
INTENT_MARGIN_DEFAULT: float = 0.0
CONTEXT_HISTORY_LIMIT_DEFAULT: int = 10

NLU_INTENT_ENGINE_DEFAULT: str = "legacy"
NLU_ENTITY_ENGINE_DEFAULT: str = "deterministic"
INTENT_INFERENCE_BUDGET_MS_DEFAULT: int = 400
ENTITY_INFERENCE_BUDGET_MS_DEFAULT: int = 400
CONTEXT_MAX_CHARS_DEFAULT: int = 4000
CONTEXT_TURNS_FOR_MODEL_DEFAULT: int = 5

NLU_INTENT_MAX_CHARS_DEFAULT: int = 0  # 0 = disabled (no extra cap)
NLU_NER_MAX_CHARS_DEFAULT: int = 0  # 0 = disabled; else skip neural NER for longer raw text

SERVER_HOST_DEFAULT: str = "0.0.0.0"
SERVER_PORT_DEFAULT: int = 8000
DEBUG_DEFAULT: bool = False
LOG_LEVEL_DEFAULT: str = "INFO"

CORS_ORIGINS_DEFAULT: List[str] = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
CORS_ALLOW_CREDENTIALS_DEFAULT: bool = True

MAX_RESULTS_DEFAULT: int = 100
MAX_SUGGESTIONS_DEFAULT: int = 20


def get_intent_threshold() -> float:
    return float(os.getenv("INTENT_THRESHOLD", INTENT_THRESHOLD_DEFAULT))


def get_context_history_limit() -> int:
    return int(os.getenv("CONTEXT_HISTORY_LIMIT", CONTEXT_HISTORY_LIMIT_DEFAULT))


def get_intent_margin_M() -> float:
    return float(os.getenv("INTENT_MARGIN", INTENT_MARGIN_DEFAULT))


def get_nlu_intent_engine() -> str:
    return os.getenv("NLU_INTENT_ENGINE", NLU_INTENT_ENGINE_DEFAULT).strip().lower()


def get_nlu_entity_engine() -> str:
    return os.getenv("NLU_ENTITY_ENGINE", NLU_ENTITY_ENGINE_DEFAULT).strip().lower()


def get_intent_threshold_T() -> float:
    return float(os.getenv("INTENT_THRESHOLD_T", os.getenv("INTENT_THRESHOLD", INTENT_THRESHOLD_DEFAULT)))


def get_intent_budget_ms() -> int:
    # Prefer design-aligned env var names; keep older names as fallback.
    return int(
        os.getenv(
            "NLU_INTENT_BUDGET_MS",
            os.getenv("INTENT_INFERENCE_BUDGET_MS", INTENT_INFERENCE_BUDGET_MS_DEFAULT),
        )
    )


def get_entity_budget_ms() -> int:
    # Prefer design-aligned env var names; keep older names as fallback.
    return int(
        os.getenv(
            "NLU_ENTITY_BUDGET_MS",
            os.getenv("ENTITY_INFERENCE_BUDGET_MS", ENTITY_INFERENCE_BUDGET_MS_DEFAULT),
        )
    )


def get_context_max_chars() -> int:
    return int(os.getenv("CONTEXT_MAX_CHARS", CONTEXT_MAX_CHARS_DEFAULT))


def get_context_turns_for_model() -> int:
    return int(os.getenv("CONTEXT_TURNS_FOR_MODEL", CONTEXT_TURNS_FOR_MODEL_DEFAULT))


def get_server_host() -> str:
    return os.getenv("SERVER_HOST", SERVER_HOST_DEFAULT)


def get_server_port() -> int:
    return int(os.getenv("SERVER_PORT", SERVER_PORT_DEFAULT))


def get_debug_mode() -> bool:
    debug_str = os.getenv("DEBUG", str(DEBUG_DEFAULT)).lower()
    return debug_str in ("true", "1", "yes", "on")


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", LOG_LEVEL_DEFAULT).upper()


def get_cors_origins() -> List[str]:
    origins_str = os.getenv("CORS_ORIGINS", None)
    if origins_str:
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    return CORS_ORIGINS_DEFAULT


def get_cors_allow_credentials() -> bool:
    allow_str = os.getenv("CORS_ALLOW_CREDENTIALS", str(CORS_ALLOW_CREDENTIALS_DEFAULT)).lower()
    return allow_str in ("true", "1", "yes", "on")


def get_max_results() -> int:
    return int(os.getenv("MAX_RESULTS", MAX_RESULTS_DEFAULT))


def get_max_suggestions() -> int:
    return int(os.getenv("MAX_SUGGESTIONS", MAX_SUGGESTIONS_DEFAULT))


def get_intent_model_path() -> str:
    # Canonical env var is NLU_INTENT_MODEL_DIR; INTENT_MODEL_PATH is a legacy alias.
    return os.getenv("NLU_INTENT_MODEL_DIR", os.getenv("INTENT_MODEL_PATH", "")).strip()


def get_ner_model_path() -> str:
    # Canonical env var is NLU_NER_MODEL_DIR; NER_MODEL_PATH is a legacy alias.
    return os.getenv("NLU_NER_MODEL_DIR", os.getenv("NER_MODEL_PATH", "")).strip()


def get_nlu_intent_max_chars() -> int:
    # 0 = disabled
    return int(os.getenv("NLU_INTENT_MAX_CHARS", NLU_INTENT_MAX_CHARS_DEFAULT))


def get_nlu_ner_max_chars() -> int:
    """If > 0 and raw message is longer, skip neural NER (deterministic only)."""
    return int(os.getenv("NLU_NER_MAX_CHARS", NLU_NER_MAX_CHARS_DEFAULT))


def get_nlu_device() -> str:
    value = os.getenv("NLU_DEVICE", "").strip().lower()
    if value:
        return "cuda" if value == "cuda" else "cpu"

    # Import torch lazily to avoid import cost at module load.
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_nlu_use_autocast() -> bool:
    raw = os.getenv("NLU_AUTOCAST", "").strip().lower()
    if raw:
        return raw in ("true", "1", "yes", "on")
    return get_nlu_device() == "cuda"
