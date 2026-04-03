# Phase 0 Research: Model-based Intent + NER

## Decision: Transformer-based intent classification

**Choice**: Use a HuggingFace `AutoModelForSequenceClassification` multi-class model for intent classification.

**How scoring works (required for fallback)**:
- Model outputs logits for all intents.
- Convert logits to probabilities via `softmax`.
- Let `top1`/`top2` be the highest and second-highest intent probabilities, and let `intent_top1` be the label of `top1`.
- Public `score` returned by the pipeline MUST be `top1`.
- Fallback MUST trigger when:
  - `top1 < T`, OR
  - `(top1 - top2) < M`

If model artifacts are missing/corrupted, the intent component must degrade to `intent="fallback"` and `score=0.0` (no crash, no schema changes).

**Rationale**:
- Directly provides ranked probabilities needed for the ambiguity-margin rule in the spec.
- Keeps the inference interface stable and mockable for tests (we can stub the probability output).

**Alternatives considered**:
- Zero-shot LLM classification (hard to mock deterministically; latency/cost; harder to guarantee schema-stable confidence semantics).
- Embedding + cosine similarity using dense embeddings (does not naturally produce calibrated `top1` probability; would still require probability calibration).
- Retaining TF-IDF + cosine similarity (explicitly out of scope for the model migration).

## Decision: Transformer-based entity extraction (NER)

**Choice**: Use a HuggingFace `AutoModelForTokenClassification` BIO tagger for entity extraction.

**Entity representation**:
- Convert token-level BIO predictions into span entities with:
  - `label`
  - `text` (surface form)
  - `start` / `end` as Unicode code point indices in the original input
  - `source="model"`

**Span indexing approach**:
- Prefer fast tokenizers with `return_offsets_mapping=True` so each predicted token has a character offset into the *original input string*.
- Convert those offsets to Python string indices (Python indices operate on Unicode code points), producing `start` inclusive and `end` exclusive.

**Fallback**:
- If entity model artifacts are missing/corrupted (and entity engine is enabled), degrade to deterministic entity extraction (patterns/dictionaries) and set `entities[].source="pattern"` or `"dictionary"` accordingly.

**Rationale**:
- BIO token classification is the most direct way to produce spans consistent with the spec.
- Offset mappings make the Unicode code point indexing requirement tractable.

**Alternatives considered**:
- Using a single-shot “extract entities as text” approach (harder to guarantee stable spans + indices).
- Keeping underthesea NER as the model replacement (explicitly conflicts with the feature intent to replace underthesea for extraction efficiency).

## Decision: Engine selection + component-wise degradation

**Choice**:
- Keep a deterministic engine for both intent and entity extraction as a safe baseline (already present in the codebase).
- Add model engines for intent and entity extraction.
- Allow operators to enable/disable intent/entity engines independently via configuration.

**Behavior**:
- Intent engine:
  - `model` enabled: run transformer inference; on artifact failure -> `fallback`.
  - `model` disabled: run legacy intent detector (TF-IDF centroid cosine) as the explicit rollout-safe backup.
- Entity engine:
  - `model` enabled: run transformer NER; on artifact failure -> deterministic patterns/dictionaries.
  - `model` disabled: run deterministic patterns/dictionaries.

**Rationale**:
- Meets spec FR-007 without requiring code changes for rollout and quick revert.
- Meets spec FR-006 “graceful degradation without crashing” per component.

## Decision: Inference time budgets

**Choice**:
- Enforce an inference time budget per component (intent and entities), and degrade gracefully if exceeded.

**Required behavior**:
- If intent budget is exceeded (or model inference errors), return `intent="fallback"` and `score=0.0` while still running deterministic entity extraction.
- If entity budget is exceeded (or model inference errors), degrade to deterministic entity extraction.

**Rationale**:
- Spec requires request-volume resilience and “do not materially degrade responsiveness”.
- Keeps the system robust under CPU pressure.

## Decision: Observability without leaking user raw text

**Choice**:
- Use the existing `logging` setup and emit structured log fields like:
  - request id (already available via middleware)
  - component (intent/entity)
  - artifact load status (loaded/missing/corrupted)
  - inference latency in milliseconds
  - fallback reason (confidence threshold vs timeout vs artifact failure)

**Rule**:
- Do not log raw user input text in production logs. Log only message-derived hashes or token counts if needed for debugging.

