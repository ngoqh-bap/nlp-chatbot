# Quickstart: Model-based Intent + NER

## 1) Install dependencies

```bash
uv sync
```

## 2) Configure model engines + fallback thresholds

The feature supports enabling/disabling intent/entity components independently.

Example environment variables (pick values to match your calibration):

```bash
# Engine selection
INTENT_ENGINE=model        # or legacy
ENTITY_ENGINE=model       # or deterministic

# Fallback thresholds
INTENT_CONFIDENCE_THRESHOLD_T=0.25   # T (top1 probability threshold)
INTENT_AMBIGUITY_MARGIN_M=0.10        # M (margin threshold)

# Inference time budgets (ms)
INTENT_INFERENCE_BUDGET_MS=1000
ENTITY_INFERENCE_BUDGET_MS=1200

# Model artifacts locations (implementation-defined; keep them consistent)
MODEL_ARTIFACTS_DIR=model_artifacts
```

> Note: Exact env var names should be aligned with the final implementation. The intent of this section is to document the operator knobs required by the spec (engine selection, `T`, `M`, and inference budgets).

## 3) Place model artifacts

Provide local artifacts under the configured directory, for example:

```text
model_artifacts/
  intent/
    config.json
    model.safetensors|pytorch_model.bin
    tokenizer.* (optional if shared)
  ner/
    config.json
    model.safetensors|pytorch_model.bin
    tokenizer.* (optional if shared)
```

If artifacts are missing/corrupted:
- intent degrades to `intent="fallback"`, `score=0.0`
- entities degrade to deterministic patterns/dictionaries

## 4) Run the backend

```bash
uvicorn main:app --reload
```

API:
- `POST /chat/advanced`
- `POST /chat/context`

## 5) Run tests

```bash
pytest
```

For model-dependent code, tests MUST use deterministic mocks (no model weight downloads required).

