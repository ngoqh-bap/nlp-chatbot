# Quickstart

Get the HUCE admission chatbot API and optional Reflex UI running on your machine.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another PEP 517 installer
- **Git**

Optional for GPU inference: CUDA-capable driver matching your PyTorch build.

## 1. Clone and install

```bash
git clone https://github.com/ngoqh-bap/nlp-chatbot.git
cd nlp-chatbot
uv sync
```

This installs backend dependencies (FastAPI, PyTorch, Transformers, etc.) and dev tools into `.venv`.

## 2. Environment

```bash
cp env.example .env
```

Edit `.env` if you need to change the API port, CORS, or NLP settings. Defaults are safe for local development.

Key variables (see `env.example` for the full list):

| Variable | Purpose |
|----------|---------|
| `SERVER_HOST` / `SERVER_PORT` | API bind (default `0.0.0.0:8000`) |
| `NLU_INTENT_ENGINE` | `legacy` (default) or `transformer` |
| `NLU_ENTITY_ENGINE` | `deterministic` (default), `transformer`, or `off` |
| `NLU_INTENT_MODEL_DIR` | Directory with a fine-tuned intent model (when using `transformer`) |
| `NLU_NER_MODEL_DIR` | Directory with a token-classification NER model (when using `transformer`) |

## 3. Run the API

From the repository root:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **API:** http://127.0.0.1:8000  
- **OpenAPI / Swagger:** http://127.0.0.1:8000/docs  

### Smoke test

```bash
curl -s http://127.0.0.1:8000/ | head -c 200
```

```bash
curl -s -X POST http://127.0.0.1:8000/chat/advanced -H "Content-Type: application/json" \
  -d '{"message":"Xin chào","session_id":"quickstart","use_context":false}'
```

On Windows PowerShell, use the same JSON in a single line or `Invoke-RestMethod`.

## 4. Run the Reflex frontend (optional)

In a **second** terminal, from the repo root:

```bash
cd frontend
uv run reflex run
```

The UI is served on **http://localhost:3000** (see `frontend/rxconfig.py`). It calls the FastAPI backend at `BACKEND_URL` (default `http://localhost:8000`).

## 5. Tests and lint

From the repository root:

```bash
uv run pytest
uv run ruff check .
```

## 6. Optional: fine-tune intent (PhoBERT)

If you want a local transformer intent model:

```bash
uv run python scripts/train_intent_phobert.py --data data/intent.csv --out models/intent/run1 --base vinai/phobert-base
```

Then set `NLU_INTENT_MODEL_DIR` to the `models/intent/run1` path, `NLU_INTENT_ENGINE=transformer`, and restart the API.

## Troubleshooting

- **`ModuleNotFoundError` for `torch` / `transformers`:** Run commands through `uv run` so the project virtual environment is used.
- **Port already in use:** Change `SERVER_PORT` in `.env` or pass `--port` to `uvicorn`.
- **Frontend cannot reach API:** Check `BACKEND_URL` in the environment when starting Reflex, and CORS settings in `.env`.
