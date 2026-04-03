# Design: Model-based preprocessing + intent + entities + efficient context

**Feature Branch**: `001-intent-ner-models`  
**Date**: 2026-04-03  
**Scope**: Replace TF‑IDF + cosine intent matching with a transformer intent engine; add optional transformer NER alongside existing pattern/dictionary extraction; keep deterministic preprocessing. Extend the existing context shaping layer (`ContextProcessor` in `nlu/context.py`) where needed for multi-turn efficiency while preserving the NLP contract.

## Goals / Non-goals

### Goals

- Replace legacy intent detection (TF‑IDF centroid cosine) with a **transformer classifier** that returns calibrated probabilities suitable for the required fallback semantics.
- Add **model-based** intent and optional NER (GPU-accelerated where available). Baseline preprocessing is already deterministic (`nlu/preprocess.py`); baseline entities are patterns/dictionaries with optional transformer NER as an add-on, not a replacement for rules.
- Preserve the public NLP schema and semantics:
  - `NLPPipeline.analyze(text) -> {"intent": str, "score": number, "entities": list}`
  - `score` is **top-1 intent probability** in `[0.0, 1.0]`
  - fallback triggers if `top1 < T` **or** `(top1 - top2) < M`
- Upgrade context handling so **multi-turn follow-ups** are interpreted correctly with bounded compute and predictable latency.
- Enforce inference budgets and graceful degradation so overall responsiveness does not materially regress.

### Non-goals

- Changing public API shapes outside the NLP output contract.
- Renaming canonical intent/entity labels (unless required for correctness).
- Introducing external online services (must work with local artifacts).

## Current state (baseline)

- **Intent**: `nlu/intent.py` implements legacy TF‑IDF + softmax-over-centroid similarities with `T`/`M`-style fallback; a transformer intent engine is planned but not yet the default path.
- **Preprocessing**: `nlu/preprocess.py` uses deterministic Unicode normalization + whitespace tokenization (no standalone ML segmentation model).
- **Entities**: `nlu/entities.py` uses patterns + dictionary phrase matching with span indices; optional transformer NER is planned.
- **Context**: `services/nlp_service.py` uses an in-memory `ContextStore` keyed by `session_id` with a history limit; stored fields include `last_intent`, `last_entities`, and `conversation_history`.

## Proposed architecture (Approach A)

Introduce explicit, pluggable engines and a context shaping layer.

### Components

- **ContextProcessor** (existing in `nlu/context.py`, used by `NLPPipeline.analyze_with_context`): converts `(message, current_context)` into bounded `model_input_text` (summary + last k turns + current message) with deterministic char budget; **extensions** in this design include structured sticky memory (Tier 1) and aligning all transformer intent paths with the same bounding rules.
- **IntentEngine** interface:
  - `LegacyTfidfIntentEngine` (existing logic)
  - `TransformerIntentEngine` (new, model-based)
- **EntityEngine** interface:
  - `DeterministicEntityEngine` (existing patterns/dictionaries)
  - `TransformerNerEngine` (new, model-based)
- **NLPPipeline**: orchestrates preprocessing/context shaping + intent + entities and returns the stable contract output.

### Data flow (per request)

1. **Context shaping**:
  - Read `current_context` (session-scoped, already stored by the service layer)
  - Build a bounded `model_input_text` for the current message using:
    - a compact “session summary” string (optional, maintained incrementally)
    - last k turns of dialogue (bounded by max turns and/or max chars)
    - the current user message
  - Also extract structured “sticky” context features (e.g., last major, year, method) from stored entities.
2. **Intent inference** (selected engine):
  - Transformer engine returns per-intent probabilities via `softmax(logits)`
  - Compute `top1`, `top2`, apply fallback rule:
    - fallback if `top1 < T` **or** `(top1 - top2) < M`
  - Apply inference time budget; on timeout/error/artifact load failure return:
    - `intent="fallback"`, `score=0.0`
3. **Entity extraction** (selected engine):
  - Deterministic extraction always available (patterns/dictionaries)
  - Transformer NER returns span entities with offsets; merge with deterministic outputs
  - Apply inference time budget; on timeout/error/artifact load failure fall back to deterministic-only
4. **Return** `{"intent": intent, "score": score, "entities": entities}` unchanged.

## Model choices

### Approved stack (2026-04-03) — GPU-first (e.g. RTX 5060)

This project uses **Approach A**: separate intent classifier + separate NER model, both **Vietnamese-capable encoder models**, with **deterministic preprocessing** (no standalone “preprocessing model”). Training may use GPU; production target is **GPU inference on a local RTX 5060-class card** with CPU fallback acceptable but not the primary SLO.


| Phase             | Model role                 | Recommended base                                                                                                                                 | Runtime API                                                                        | Notes                                                                                                                                           |
| ----------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Preprocessing     | None (deterministic)       | N/A                                                                                                                                              | `normalize_text` + tokenizer from HF models only where needed                      | Avoid extra segmentation models; encoders handle subwords.                                                                                      |
| Intent            | Multi-class classification | **PhoBERT-family** encoder (e.g. `vinai/phobert-base` or `vinai/phobert-large-v2` as capacity/latency trade-off) fine-tuned on `data/intent.csv` | `AutoModelForSequenceClassification` + `AutoTokenizer(use_fast=True)`              | `softmax` → `top1`/`top2`; drives `T`/`M`. Prefer **base** first for latency; scale to **large-v2** if accuracy needs it and P95 budget allows. |
| Entities (neural) | Token classification (BIO) | **PhoBERT-based** NER checkpoint (community Vietnamese NER on PhoBERT, or fine-tune PhoBERT on your label set)                                   | `AutoModelForTokenClassification` + fast tokenizer + `return_offsets_mapping=True` | Run on **raw user message** only; merge with pattern/dictionary.                                                                                |
| Entities (rules)  | Deterministic              | N/A                                                                                                                                              | Existing pattern + dictionary extractors                                           | Always-on safety net; `source` is `pattern` / `dictionary`.                                                                                     |


**RTX 5060 inference guidance (non-binding defaults for planning):**

- Use **CUDA** (`torch` device `cuda`), **FP16 or BF16** autocast where numerically stable for throughput; FP32 acceptable for debugging.
- **Batch size 1** for chat latency; keep separate CUDA streams only if measured benefit.
- **VRAM headroom**: PhoBERT-base + PhoBERT-base NER typically fit comfortably; if both models are resident simultaneously, prefer **smaller checkpoints** or **load one model at a time** (intent then NER) if memory pressure appears—measure on hardware.
- **Warm-up**: optional one forward pass at process start to stabilize latency.

### Intent classifier

- HuggingFace `AutoModelForSequenceClassification` with a **PhoBERT** backbone (fine-tuned), unless offline eval shows insufficient quality.
- Output semantics:
  - logits -> probabilities via `softmax`
  - `score` returned to callers is `top1` probability (not cosine similarity)

### NER

- HuggingFace `AutoModelForTokenClassification` (BIO tagging) with a **PhoBERT** backbone (fine-tuned or a compatible pretrained NER head).
- Span reconstruction:
  - Use a fast tokenizer with `return_offsets_mapping=True` to map token predictions back to character offsets in the original input
  - Produce entities with:
    - `label`
    - `text`
    - `start` / `end` as Python string indices (Unicode code points), start inclusive / end exclusive
    - `source="model"`

**Offset contract (required)**:

- `start`/`end` MUST be Unicode code point indices into the **raw current user message text passed to** `NLPPipeline.analyze(text)`.
- Offsets MUST NOT be relative to any combined `model_input_text` that includes prior turns, summaries, or prefixes (e.g., `"User: "`).
  - To keep offsets correct and implementation simple, `TransformerNerEngine` runs on the **raw current message only** (context is used for intent, not for span extraction).

## Two-tier context processing (efficiency upgrade)

The goal is to use “the entire current session” conceptually, while keeping compute bounded and predictable.

### Tier 1: Structured memory (fast path)

Maintain small, structured context fields in `ContextStore`:

- `last_intent`: previous resolved intent label
- `last_entities`: previous entities (already present)
- `sticky_entities`: a compact mapping of important entity types (e.g., major, year, admission method) to the most recent value + timestamp/turn index

Structured memory is:

- cheap to update
- cheap to read
- robust when conversation history is truncated

### Tier 2: Bounded textual context (model input)

Maintain a bounded textual representation for model consumption:

- `conversation_history`: keep the last `HISTORY_LIMIT` entries (existing behavior)
- `context_summary`: an optional rolling summary string that is updated when history would exceed the model input budget

ContextProcessor builds model input as:

```text
<optional context_summary>
<last k turns (bounded by max chars)>
User: <current message>
```

Budget controls:

- `CONTEXT_MAX_CHARS` (or `CONTEXT_MAX_TOKENS`)
- `HISTORY_LIMIT` (already exists via config; tests expect default 10)
- `CONTEXT_TURNS_FOR_MODEL` (k)

This design makes long sessions safe: history can grow conceptually via summary + structured memory, but model input remains bounded.

### Determinism + bounding rules (required)

- `conversation_history` retention MUST continue to respect `HISTORY_LIMIT` (default 10 as in current behavior/tests).
- `ContextProcessor` MUST build `model_input_text` deterministically under `CONTEXT_MAX_CHARS` (or tokens).
- `context_summary` MUST be deterministic (no stochastic summarization model). Suggested default:
  - When `model_input_text` would exceed the budget, move the oldest turns into `context_summary` by appending a compact, deterministic representation (e.g., truncated user message + selected structured memory such as last intent/entities labels), then drop those turns from the textual window.

## Configuration / rollout knobs

Environment/config keys (names indicative; final names should match existing config conventions):

- `NLU_INTENT_ENGINE=legacy|transformer`
- `NLU_ENTITY_ENGINE=deterministic|transformer|off`
- `NLU_INTENT_T=<float>` (confidence threshold)
- `NLU_INTENT_M=<float>` (ambiguity margin)
- `NLU_INTENT_BUDGET_MS=<int>`
- `NLU_ENTITY_BUDGET_MS=<int>`
- `CONTEXT_HISTORY_LIMIT=<int>`
- `CONTEXT_MAX_CHARS=<int>` (or tokens)
- `CONTEXT_TURNS_FOR_MODEL=<int>`

Rollout strategy:

- default to legacy/deterministic until offline evaluation validates the transformer engines
- allow independent toggling and instant rollback without code changes

`NLU_ENTITY_ENGINE=off` means: skip entity extraction and return `entities=[]` (while still returning a valid schema).

## Degradation and error handling (required)

Component-wise degradation rules:

- Intent engine failures MUST NOT crash entity extraction.
- Entity engine failures MUST NOT crash intent classification.

Specific behavior:

- **Intent artifact missing/corrupted** or inference error/timeout:
  - return `intent="fallback"`, `score=0.0`
- **Entity artifact missing/corrupted** or inference error/timeout:
  - return deterministic entities only; each entity MUST still conform to `{label, text, start, end, source}` with `source ∈ {"pattern","dictionary"}`

## Edge cases (required; aligns with `specs/001-intent-ner-models/spec.md`)

- **Empty or symbols-only input**: return `intent="fallback"`, `score=0.0`, `entities=[]`.
- **Extremely long input**:
  - Truncate to a safe maximum for intent inference (bounded compute).
  - If inference succeeds on the truncated text, apply the normal fallback rule (`T`/`M`) and return `score=top1` even when the returned intent is `"fallback"`.
  - Regardless of truncation, still run deterministic entity extraction on the original message text.
  - If intent inference times out/fails, return `intent="fallback"`, `score=0.0`.
- **Mixed-language / slang / typos / near-ties**: prefer safe fallback by applying `T`/`M` (low top-1 confidence or small top-1/top-2 margin triggers fallback).

## Observability / logging

- Emit structured logs with:
  - request id
  - component (`intent`, `entities`, `context`)
  - engine selection
  - latency metrics
  - fallback reason (`threshold`, `margin`, `timeout`, `artifact_missing`, `error`)
- Do NOT log raw user message text in production logs.

## Testing strategy

- Unit tests for:
  - fallback semantics (`T`, `M`, `top1`, `top2`)
  - time budget -> graceful degradation paths
  - entity span reconstruction with offsets (Unicode code points)
  - ContextProcessor bounding behavior (history + summary + structured memory)
- Integration tests for:
  - `/chat/advanced` output schema stability
  - engine toggles

## Open questions (to resolve during implementation planning)

- **Exact checkpoint IDs** to pin (intent + NER): choose between `phobert-base` vs `phobert-large-v2` after a short latency/accuracy bake-off on RTX 5060.
- **Artifact layout**: e.g. `models/intent/<run-id>/` and `models/ner/<run-id>/` with `config.json`, tokenizer, weights; env vars `INTENT_MODEL_PATH` / `NER_MODEL_PATH` (names TBD in `config.py`).
- Whether to implement `context_summary` as extractive (e.g., last facts) or as a deterministic compression heuristic (to keep tests deterministic and avoid dependency on a summarization model).
- Exact config key names to align with `config.py` (partially started: engine toggles and budgets).

