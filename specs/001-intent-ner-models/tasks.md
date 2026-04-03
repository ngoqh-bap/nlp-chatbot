# Tasks: Model-based Intent + NER

**Input**: Design documents from `/specs/001-intent-ner-models/`
**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories), `research.md`, `data-model.md`, `contracts/`
**Tests**: This feature’s spec includes acceptance scenarios with deterministic testability requirements; therefore this plan includes focused pytest tasks (no real model weight downloads in tests).
**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Backend code lives in `nlu/`, `services/`, and root FastAPI entrypoint `main.py`
- Tests live in `tests/` (unit + integration)

---
## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and shared infrastructure for model-based intent + entity extraction, including config, timeouts, and deterministic offset computation.

- [ ] T001 Create model backend interfaces/types in `nlu/model_backends.py`
- [ ] T002 [P] Add engine-selection and inference-budget config getters in `config.py` (`get_intent_engine`, `get_entity_engine`, `get_intent_confidence_threshold_T`, `get_intent_ambiguity_margin_M`, `get_intent_inference_budget_ms`, `get_entity_inference_budget_ms`, `get_model_artifacts_dir`)
- [ ] T003 [P] Update operator environment variables in `env.example` to match the spec (`INTENT_ENGINE`, `ENTITY_ENGINE`, `T`, `M`, and inference budgets) and document model artifacts directory (`MODEL_ARTIFACTS_DIR`)
- [ ] T004 [P] Update dependencies in `pyproject.toml` to add `transformers` and `torch` and (if targeting full migration) remove reliance on `underthesea`
- [ ] T005 [P] Update Vietnamese tokenization in `nlu/preprocess.py` to remove hard dependency on `underthesea` (keep deterministic fallback tokenizer)
- [ ] T006 [P] Create `nlu/time_budget.py` providing a timeout utility for synchronous model inference (thread-based timeout) and a typed exception for timeouts
- [ ] T007 [P] Create `nlu/offsets.py` with helpers to compute Unicode code point `start`/`end` offsets for deterministic entity spans (no raw-text logging; offsets must be computed on the original input string)
- [ ] T008 Update deterministic entity extraction in `nlu/entities.py` to return entity dicts with `{label, text, start, end, source}` where `source` is only `"pattern"` or `"dictionary"` (remove any `"ner"` source path from deterministic mode)
- [ ] T009 Create model artifact loader helpers in `nlu/model_artifacts.py` that validate local filesystem artifacts for intent and NER and raise clear, typed errors for missing/corrupted artifacts
- [ ] T010 Create engine factories/selector in `nlu/engine_selector.py` that instantiate the correct intent/entity backends based on `INTENT_ENGINE` and `ENTITY_ENGINE`, with safe defaults so unit tests run without model weights

---
## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement the stable orchestration layer that applies component-wise fallback semantics, enforces inference time budgets, and preserves the public NLP schema exactly.

**Checkpoint**: Foundation ready — user stories can now be implemented against the new orchestration behavior.

- [ ] T011 Implement component-wise fallback + schema stability in `nlu/pipeline.py` so `NLPPipeline.analyze(text)` returns exactly `{"intent": str, "score": number, "entities": list}` and each entity includes `start/end/source`
- [ ] T012 Integrate inference time budget enforcement in `nlu/pipeline.py` (intent and entity) using `nlu/time_budget.py`, ensuring timeouts degrade to the spec’s fallback behavior
- [ ] T013 Update fallback routing logic in `services/nlp_service.py` to rely on `analysis["intent"] == "fallback"` (avoid double-thresholding via the legacy `INTENT_THRESHOLD` since the pipeline will own `T`/`M` fallback semantics)
- [ ] T014 Sanitize production logging in `main.py` to avoid logging raw user text (keep request id and log safe metadata only)
- [ ] T015 Ensure `nlu/pipeline.py` applies truncation for intent inference (configurable max length or conservative safe cap) before model calls and still runs deterministic entity extraction on fallback
 - [ ] T037 Add unit tests for extremely long inputs and time-budget expirations in `tests/unit/test_intent_time_budget_and_truncation.py`, covering the edge-case behaviors defined in `spec.md` (truncation, low-confidence fallback, timeout fallback with `score=0.0`)

---
## Phase 3: User Story 1 - Better intent understanding (Priority: P1) 🎯 MVP

**Goal**: When `INTENT_ENGINE=model`, intent classification uses a transformer model with probability-based fallback using operator-controlled `T` (confidence threshold) and `M` (ambiguity margin), returning `intent="fallback"` and `score=0.0` on model artifact failures/timeouts (while still keeping entities extraction functional), and can leverage the entire current chat session as context (until reset) to better understand user intent.

**Independent Test**: With `INTENT_ENGINE=model` and injected/stubbed model outputs (no real weights), verify top1/top2 probability semantics and fallback triggers (`top1 < T` OR `(top1 - top2) < M`) plus artifact-missing degradation to `intent="fallback"` and `score=0.0`. Additionally, verify that, for multi-turn conversations within a single session, the system can correctly route intents that depend on earlier turns (and that resetting the session clears prior context).

### Tests for User Story 1 ⚠️

No real weight downloads: use deterministic stubs/monkeypatching of the transformer intent backend.

- [ ] T016 [US1] Add unit tests for model-intent fallback rules (top1<T and margin<M) in `tests/unit/test_intent_model_fallback.py`
- [ ] T017 [US1] Add unit test for missing/corrupted intent artifacts while `INTENT_ENGINE=model` (expect `intent="fallback"`, `score=0.0`) in `tests/unit/test_intent_model_artifact_failure.py`
 - [ ] T042 [US1] Add tests for multi-turn context understanding (using entire current session until reset) in `tests/integration/test_multi_turn_context_intent.py`, covering cases where the correct intent depends on earlier user turns and confirming behavior when the session is reset

### Implementation for User Story 1

- [ ] T018 [US1] Implement transformer intent backend in `nlu/model_intent.py` using `transformers` (local artifacts only), returning top1/top2 probabilities derived from logits via softmax
- [ ] T019 [US1] Wire `nlu/model_intent.py` into `nlu/engine_selector.py` and ensure `nlu/pipeline.py` calls it when `INTENT_ENGINE=model`, applying `T`/`M` fallback semantics
- [ ] T020 [US1] Add safe exception mapping in `nlu/pipeline.py` so model load/inference errors/timeouts map to `intent="fallback"` and `score=0.0` (while still extracting deterministic entities)
- [ ] T021 [US1] Update existing intent unit tests in `tests/unit/test_intent.py` to run against legacy defaults (so they don’t require model weights) and to align with the new `services/nlp_service.py` fallback routing behavior
 - [ ] T043 [US1] Implement or extend a session/context management component in `services/` (e.g., `services/session_context.py`) that maintains conversation history for the current session and exposes a function to construct context-aware text for `NLPPipeline.analyze(text)`
 - [ ] T044 [US1] Update `services/nlp_service.py` (and any relevant FastAPI handlers) to use the session context component from T043 so that `/chat/advanced` passes context-aware text based on the entire current session to the NLP pipeline while preserving the existing NLP API contract and schema

---
## Phase 4: User Story 2 - More accurate entities without slow NLP (Priority: P2)

**Goal**: When `ENTITY_ENGINE=model`, entity extraction uses a transformer token-classification NER backend to produce span entities with `start/end` in Unicode code points and `source="model"`. If entity artifacts are missing/corrupted or inference times out, entities degrade to deterministic patterns/dictionaries with `source="pattern"` / `"dictionary"` (and still produce valid `start/end`).

**Independent Test**: With `ENTITY_ENGINE=model` and injected/stubbed model outputs (no real weights), verify:
- entity span aggregation and offset conversion (`start/end`) are consistent
- timeouts and artifact failures degrade independently to deterministic extraction while preserving schema and correct `source`.

### Tests for User Story 2 ⚠️

- [ ] T022 [US2] Implement unit tests for transformer entity spans and `source="model"` with mocked offset mappings in `tests/unit/test_entity_model_spans.py`
- [ ] T023 [US2] Implement unit tests for model artifact failure while `ENTITY_ENGINE=model` (expect deterministic entities with `source="pattern"`/`"dictionary"`) in `tests/unit/test_entity_fallback_deterministic.py`
- [ ] T024 [US2] Update entity fixtures and assertions to require `start/end` and to remove any acceptance of `"ner"` as a `source` in `tests/conftest.py` and `tests/unit/test_entities.py`

### Implementation for User Story 2

- [ ] T025 [US2] Implement transformer NER backend in `nlu/model_entities.py` (`AutoModelForTokenClassification`) using `return_offsets_mapping=True` and convert offsets into Python Unicode code point indices
- [ ] T026 [US2] Wire `nlu/model_entities.py` into `nlu/engine_selector.py` and ensure `nlu/pipeline.py` selects it when `ENTITY_ENGINE=model`
- [ ] T027 [US2] Ensure component-wise entity degradation in `nlu/pipeline.py`: on model artifact/inference failure, fall back to deterministic extraction (patterns/dictionaries) and set `source` correctly
- [ ] T028 [US2] Refine deterministic entity offset handling in `nlu/entities.py` to deduplicate entities without losing valid `start/end` boundaries

---
## Phase 5: User Story 3 - Safe rollout and backwards-compatible behavior (Priority: P3)

**Goal**: Operators can enable/disable intent/entity engines independently via configuration so the API continues returning the stable NLP schema (`intent`, `score`, `entities`) while switching behavior as intended.

**Independent Test**: Toggle `INTENT_ENGINE` and `ENTITY_ENGINE` (with stubbed backends where needed) and verify:
- schema keys stay identical
- disabling the model engines still returns valid deterministic outputs
- enabling model engines uses model paths (via stubs) and preserves schema stability.

### Tests for User Story 3 ⚠️

- [ ] T029 [US3] Add unit tests for engine toggle combinations and schema stability in `tests/unit/test_engine_toggle_schema.py`
- [ ] T030 [US3] Add API integration test assertions ensuring `/chat/advanced` still returns `analysis.intent`, `analysis.score`, and `analysis.entities` keys under engine toggles in `tests/integration/test_api.py`

### Implementation for User Story 3

- [ ] T031 [US3] Ensure `nlu/engine_selector.py` reads `INTENT_ENGINE` and `ENTITY_ENGINE` exclusively from config/env and applies correct defaults for non-model environments in `nlu/engine_selector.py`
- [ ] T032 [US3] Verify end-to-end routing uses `intent="fallback"` semantics consistently in `services/nlp_service.py` and does not depend on legacy `INTENT_THRESHOLD` for model-based fallback

---
## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Bring the repository into a clean, secure, and operator-aligned state (docs, logging, and correctness checks).

- [ ] T033 Documentation alignment: update `specs/001-intent-ner-models/quickstart.md` to match the implemented env var names and artifact directory expectations
 - [ ] T034 Logging alignment (Constitution Principle II): add structured log fields for component (`intent`/`entities`), artifact status (`loaded/missing/corrupted`), inference latency, fallback reason, and (where applicable) context-window behavior (e.g., number of turns/tokens used, truncation decisions) in `nlu/pipeline.py` and `nlu/model_intent.py` / `nlu/model_entities.py` using the standard library `logging` module (no raw user text)
 - [ ] T035 Add a quick “smoke verification” test or script task to run the unit suite without model weights (ensuring stubs/default engines are used) in `tests/` (e.g., `tests/unit/test_smoke_no_model_weights.py`), satisfying the constitution requirement that model-dependent code is fully mockable
 - [ ] T036 Run formatting/lint/type checks and fix any issues introduced by the new modules using the existing `pytest`, `ruff`, and `mypy` workflow so that CI remains green in line with the constitution
 - [ ] T038 Create a held-out intent evaluation dataset and baseline/intent-model comparison script (e.g., under `tests/fixtures/intent_eval/` and `scripts/eval_intent_models.py`) to measure accuracy/F1 and wrong-intent rates, mapping to Success Criteria SC-001 and SC-002
 - [ ] T039 Calibrate default thresholds `T` and `M` using the evaluation results from T038 and document chosen values plus rationale in `specs/001-intent-ner-models/research.md` (or a short evaluation report), keeping config defaults aligned with SC-001 and SC-002
 - [ ] T040 Define and maintain a “golden set” of `/chat/advanced` requests and expected NLP outputs in `tests/fixtures/golden_chat/` and add regression tests in `tests/integration/test_golden_chat_schema_stability.py` to enforce SC-003 (schema compatibility and stability across runs)
 - [ ] T041 Implement a lightweight performance/regression check for `/chat/advanced` (e.g., a small load/latency script or test harness) that verifies the P95 latency/error budget defined in `spec.md` FR-008/SC-004 under representative normal load

---
## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — blocks all user stories
- **User Stories (Phase 3+)**: Depends on Phase 2 completion
- **Polish (Final Phase)**: Depends on completion of desired user stories

### User Story Completion Order

1. **User Story 1 (P1)**: MVP intent model + fallback correctness
2. **User Story 2 (P2)**: entity model spans + deterministic fallback offsets
3. **User Story 3 (P3)**: engine toggles and schema stability in end-to-end flows

### Parallel Opportunities

- Phase 1 tasks marked `[P]` can run in parallel (different files: config/env, dependency pins, tokenizer removal, timeout helper, offsets helper)
- After Phase 2 completes, User Story 1 and User Story 2 can be implemented in parallel if team staffing allows, but be careful not to touch the same orchestration file at the same time (`nlu/pipeline.py`)

---
## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (interfaces/config/timeouts/offset helpers + deterministic entities offsets)
2. Complete Phase 2 (pipeline orchestration + fallback routing + safe logging)
3. Complete Phase 3 (intent transformer backend + fallback tests with stubs)
4. Validate User Story 1 independently via `tests/unit/test_intent_model_fallback*.py`

### Incremental Delivery

1. Add User Story 2 (NER spans + fallback deterministic offset validation)
2. Add User Story 3 (engine toggles; schema stability through `/chat/advanced`)

