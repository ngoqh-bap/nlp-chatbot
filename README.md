# HUCE Chatbot — Admission Q&A (HUCE)

**Tiếng Việt:** Chatbot tra cứu thông tin tuyển sinh Đại học Xây dựng Hà Nội (HUCE), với NLP tiếng Việt và API FastAPI.

This repository contains a **FastAPI** backend that routes Vietnamese user messages through an **NLU pipeline** (intent + entities), then answers from curated **CSV** admission data. An optional **Reflex** frontend in `frontend/` talks to the same API.

---

## Features

- **Admissions Q&A:** Benchmark scores by major and year, tuition, scholarships, majors, admission methods, timelines, contact info.
- **NLU**
  - **Intent:** Legacy TF‑IDF + calibrated softmax scores and threshold/margin fallback; optional **PhoBERT** sequence classifier via `transformers` when `NLU_INTENT_ENGINE=transformer` and a local model directory is set.
  - **Entities:** Pattern + dictionary extraction with character spans; optional **token-classification NER** on the raw user message when `NLU_ENTITY_ENGINE=transformer`.
  - **Context:** Session history (bounded), context shaping for multi-turn intent, optional **sticky entity** map for follow-ups.
- **Safety:** Input sanitization, rate limiting hooks, structured errors.

---

## Tech stack

| Layer | Stack |
|-------|--------|
| API | Python 3.13+, FastAPI, Pydantic v2, Uvicorn |
| NLU | `nlu/` pipeline; optional `torch` + `transformers` (PhoBERT-class models) |
| Data | CSV under `data/`, loaded via `services/` |
| UI (optional) | Reflex (`frontend/`), HTTP/WebSocket to the API |
| Quality | pytest, ruff |

---

## Repository layout

```text
nlp-chatbot/
├── main.py                 # FastAPI app
├── config.py               # Environment-driven settings
├── models.py               # Pydantic request/response models
├── nlu/
│   ├── pipeline.py         # analyze / analyze_with_context
│   ├── intent.py           # Legacy intent detector + adapter
│   ├── entities.py         # Pattern + dictionary entities
│   ├── context.py          # Bounded context for intent text
│   └── engines/            # Pluggable intent/NER engines + budgets
├── services/               # NLP facade, CSV handlers, processors
├── data/                   # CSV knowledge base
├── scripts/
│   ├── train_intent_phobert.py   # Optional PhoBERT intent fine-tune
│   └── train_ner_phobert.py      # Optional PhoBERT BIO NER fine-tune (JSON Lines)
├── frontend/               # Reflex app (optional)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   └── QUICKSTART.md       # Step-by-step local setup
└── env.example             # Copy to .env
```

---

## Quickstart

**See [docs/QUICKSTART.md](docs/QUICKSTART.md)** for install, `.env`, running the API and Reflex, smoke tests, and optional intent training.

Short version:

```bash
uv sync
cp env.example .env
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

---

## Configuration

Copy `env.example` to `.env`. Important groups:

- **Server:** `SERVER_HOST`, `SERVER_PORT`, `DEBUG`, `LOG_LEVEL`, CORS.
- **NLU:** `NLU_INTENT_ENGINE`, `NLU_ENTITY_ENGINE`, `NLU_INTENT_MODEL_DIR`, `NLU_NER_MODEL_DIR`, `NLU_DEVICE`, budgets, `NLU_INTENT_MAX_CHARS`, `NLU_NER_MAX_CHARS`.

Defaults keep **legacy** intent and **deterministic** entities so the API runs without downloading models.

---

## Testing and lint

```bash
uv run pytest
uv run ruff check .
```

---

## API

Interactive docs: **`GET /docs`** (Swagger UI) when the server is running.

Typical chat request:

```http
POST /chat/advanced
Content-Type: application/json

{
  "message": "Điểm chuẩn ngành Kiến trúc?",
  "session_id": "user-1",
  "use_context": true
}
```

Response includes `analysis` (`intent`, `score`, `entities`), `response`, and `context`.

---

## Contributing

1. Use **Python 3.13+** and run tests with **`uv run pytest`** before pushing.
2. Follow existing style; **`uv run ruff check .`** should pass.
3. Prefer focused commits and PRs that describe behavior changes in plain language.

---

## License

Add a `LICENSE` file at the repository root if you redistribute this code; there is no default license in this tree.
