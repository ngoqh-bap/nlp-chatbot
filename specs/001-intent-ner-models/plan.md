# Implementation Plan: [FEATURE]

**Branch**: `001-intent-ner-models` | **Date**: 2026-04-03 | **Spec**: `spec.md`
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Replace the current intent matching (TF-IDF + cosine similarity) and Vietnamese tokenization/NER dependency with transformer-based components, while keeping the public NLP response schema stable: `{"intent": <string>, "score": <number>, "entities": <list>}`.

Implementation uses operator-selectable engines (`intent`: `model` vs `legacy`, `entities`: `model` vs `deterministic`) and component-wise graceful degradation aligned with `spec.md` FR-004 and Edge Cases:
- Intent falls back to `"fallback"` according to the confidence/ambiguity thresholds `T` and `M` defined in the spec (including artifact-missing and timeout scenarios where `score = 0.0`).
- Entity extraction falls back to deterministic pattern/dictionary extraction when the entity model is unavailable (and must set `entities[].source` to a documented deterministic value such as `"pattern"` or `"dictionary"`).
The NLP pipeline must also cooperate with improved context management so that intent and entities can be interpreted using **the entire current chat session** (until the user explicitly resets the session), while still presenting the same single-message NLP API contract to callers.
Both model paths are constrained by inference time budgets to avoid unacceptable responsiveness regressions.

## Technical Context
<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13+  
**Primary Dependencies**: FastAPI + pydantic v2 + transformers + torch (model inference) + Reflex (frontend)  
**Storage**: N/A for this feature (model artifacts on local filesystem under implementation-defined directory)  
**Testing**: pytest (incl. deterministic mocks) + pytest-asyncio (if needed) + ruff (lint) + mypy (type checks)  
**Target Platform**: Web server (CPU-supported; GPU optional)  
**Project Type**: Full-stack web app (Reflex frontend + FastAPI backend)  
**Performance Goals**: Enforce `INTENT_INFERENCE_BUDGET_MS` and `ENTITY_INFERENCE_BUDGET_MS`; if exceeded, degrade gracefully without schema changes.  
**Constraints**: Operator-configurable thresholds `T` and ambiguity margin `M`; component-wise fallback behavior; do not log raw user text in production logs.  
**Scale/Scope**: Typical single-message chat traffic with context derived from the current session; must remain responsive under increased request volume via time budgets and graceful degradation, even when session histories are long (implementation may apply internal context-window limits to satisfy FR-008/SC-004 while preserving behavior).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Gates (from `constitution.md`):
- Non-Negotiable Test Coverage: all acceptance scenarios in `spec.md` must map to `tests/` with deterministic fixtures; model-dependent inference must be mockable (no weight downloads / no GPU requirement for tests).
- Respect the current product contract: `NLPPipeline.analyze(text) -> {"intent","score","entities"}` schema shape is preserved exactly.
- Pythonic AI-First Code: type hints on public interfaces, concise “why” comments where span/label normalization and fallback logic are non-obvious, and structured logging (without raw input text).
- Complexity/Dependency governance: `transformers` + `torch` usage must remain within the approved AI/ML stack; any additional library/tooling beyond that requires explicit justification (recorded in this plan if it becomes necessary).

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
frontend/
└── chatbot/

services/
├── handlers/
└── processors/
  (plus service modules like `nlp_service.py`, `csv_service.py`)

nlu/
├── pipeline.py
├── intent.py
├── entities.py
└── preprocess.py

tests/
├── fixtures/
├── integration/
└── unit/
```

**Structure Decision**: Keep NLP orchestration within `nlu/` and service integration (including session/context management and passing conversation history into the NLP pipeline) within `services/`, with runtime entrypoints exposed via `main.py` and test coverage under `tests/` (unit + integration). `frontend` remains a thin UI layer that drives API calls and does not own NLP or context logic.

## Complexity Tracking

> No additional constitution violations requiring justification are expected for Phase 0/1 design, assuming `transformers` + `torch` remain within the approved AI/ML stack and tests fully mock model inference.

