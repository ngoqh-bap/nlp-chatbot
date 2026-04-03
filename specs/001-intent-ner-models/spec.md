# Feature Specification: Model-based Intent + NER

**Feature Branch**: `001-intent-ner-models`  
**Created**: 2026-04-01  
**Status**: In Review  
**Input**: User description: "replace tfidf and cosine similarity into using models, and replace underthesea to more efficiency models"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Better intent understanding (Priority: P1)

As a prospective student (or parent), I want the chatbot to correctly understand
what I’m asking (my “intent”) even when my wording differs from the examples or
depends on earlier turns in the conversation (within the current session), so I
get the right answer path without retries.

**Why this priority**: Intent routing is the first decision the chatbot makes;
misclassification sends users to the wrong answer logic and breaks trust.

**Independent Test**: Provide a fixed set of representative messages and verify
the system returns the expected intent label (or a safe fallback when uncertain)
without changing the public response schema.

**Acceptance Scenarios**:

1. **Given** the system is running with the new intent engine enabled, **When**
   a user sends a message that is semantically equivalent to a known intent but
   uses different wording, **Then** the system returns the correct intent label
   with a confidence score.
2. **Given** the system is running with the new intent engine enabled, **When**
   a user sends an ambiguous message, **Then** the system returns the configured
   fallback intent and includes the confidence score that triggered fallback.
   Fallback is triggered when either (a) the top intent probability is below the
   configured confidence threshold `T`, or (b) the ambiguity margin `(top1 - top2)`
   is below the configured margin `M` (with `score` set to `top1`). `T` and `M`
   are operator-configurable; defaults are chosen by offline calibration on the
   held-out evaluation set (operator override allowed).
3. **Given** the intent engine cannot load (e.g., missing runtime artifact),
   **When** a user sends a message, **Then** the system still responds and routes
   safely using fallback behavior (no crashes and no schema changes).
4. **Given** the system is running with the new intent engine enabled and the
   user is in an ongoing session, **When** the user asks a follow-up question
   whose correct intent depends on an earlier turn in the same session, **Then**
   the system uses the entire current session as context (until reset) to select
   the correct intent and returns it with a confidence score.
5. **Given** the user explicitly resets the session, **When** they send a new
   message afterward, **Then** the system treats it as a fresh conversation
   without using pre-reset context, while still applying the same intent engine
   and fallback semantics.

---

### User Story 2 - More accurate entities without slow NLP (Priority: P2)

As a user, I want the chatbot to correctly extract key details from my message
(e.g., major name, year) so the answer is specific and relevant, without the
system becoming slow or fragile due to heavy NLP dependencies.

**Why this priority**: Correct entities are needed to answer questions like
admission scores by year/major; extraction quality directly affects usefulness.

**Independent Test**: For a fixed set of messages, verify the entities list
contains the expected extracted items and keeps the same output schema, while
the system remains responsive.

**Acceptance Scenarios**:

1. **Given** the system is running with the new entity extraction enabled,
   **When** a user message contains recognizable details, **Then** the response
   includes an `entities` list containing those details with labels and text
   spans represented as `start`/`end` Unicode code point indices, plus `source` indicating
   which extraction engine produced the entity.
2. **Given** the system is running with the new entity extraction enabled,
   **When** the message contains no extractable details, **Then** the response
   includes an empty entities list (not an error).
3. **Given** the advanced entity extraction is unavailable (e.g., required runtime
   artifact missing),
   **When** a user sends a message, **Then** the system still extracts entities
   using existing deterministic rules (patterns/dictionaries) and responds.

---

### User Story 3 - Safe rollout and backwards-compatible behavior (Priority: P3)

As an operator/maintainer, I want to enable or disable the new intent and entity
engines without code changes, so we can roll out gradually and quickly revert if
quality or performance is not acceptable.

**Why this priority**: This reduces deployment risk for core NLP behavior and
helps validate improvements incrementally.

**Independent Test**: Toggle configuration and verify responses keep the same
schema while switching behavior as intended.

**Acceptance Scenarios**:

1. **Given** configuration supports selecting the intent engine,
   **When** the operator switches between engines, **Then** the system continues
   to respond with the same fields (`intent`, `score`, `entities`).
2. **Given** configuration supports selecting the entity extraction engine,
   **When** the operator disables the new extraction engine, **Then** deterministic
   extraction still runs and the response remains valid.

### Edge Cases

- What happens when the input is empty, extremely long, or contains only symbols?
  - Empty/symbols-only: return `"fallback"` with `score = 0.0` and `entities = []`.
  - Extremely long: truncate to a safe maximum length for intent inference, then:
    - if intent inference succeeds but confidence is below the configured threshold:
      return `"fallback"` (with the confidence score that triggered fallback)
    - if intent inference fails or times out:
      return `"fallback"` (with `score = 0.0`)
    - in all cases: still run deterministic entity extraction (patterns/dictionaries)
      and preserve the normal response schema.
- What happens when input contains mixed Vietnamese/English or heavy slang/typos?
  - Mixed-language/slang/typo-heavy messages MUST prefer a safe fallback over confidently wrong intents: if the model’s top intent probability is below `T` **or** `(top1 - top2) < M`, return `"fallback"` rather than routing to a specific intent.
- What happens when a message could match multiple intents equally well?
  - Let `p1` be the top-1 intent probability and `p2` be the top-2 intent probability.
    Set `score = p1`. Return `fallback` if `score < T` **or** `(p1 - p2) < M`.
    (If fallback is triggered due to timeout/failure, `score` follows the failure
    behavior below, e.g. `0.0`.)
- What happens when runtime artifacts required for improved NLP are missing/corrupted?
  - Degrade independently per component:
    - If the intent model artifacts are missing/corrupted (and intent engine is enabled),
      return `intent = "fallback"` with `score = 0.0`.
    - If the entity model artifacts are missing/corrupted (and entity extraction engine is enabled),
      fall back to deterministic entity extraction (patterns/dictionaries) and set
      `entities[].source = "pattern"`.
    - If both intent and entity artifacts are missing/corrupted, apply both fallbacks.
- What happens when the system runs on a lower-performance environment?
- How does the system prevent timeouts under high request volume?
  - Enforce an inference time budget for model-based engines; if the budget is exceeded,
    degrade gracefully by returning `"fallback"` for intent (score `0.0`) while still
    running deterministic entity extraction and preserving the response schema.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST replace the current intent-matching approach with an intent engine that improves understanding of varied phrasing.
- **FR-002**: System MUST replace the current Vietnamese tokenization/NER dependency with a more efficient approach for intent and entity extraction.
- **FR-003**: System MUST preserve the public NLP response schema: `{"intent": <string>, "score": <number>, "entities": <list>}`.
- **FR-004**: System MUST support a low-confidence fallback intent behavior that avoids incorrect routing when uncertain. If the model-based intent inference succeeds, `score` MUST represent the top intent probability (`top1`), and fallback MUST be triggered when `top1 < T` **or** `(top1 - top2) < M`. `T` and `M` are operator-configurable; defaults are chosen by offline calibration on the held-out evaluation set (operator override allowed).
- **FR-005**: System MUST continue to extract entities using existing deterministic rules (patterns/dictionaries).
- **FR-006**: System MUST continue to operate if model components are unavailable (graceful degradation without crashing).
  Model components MUST degrade independently: intent artifacts missing => `intent="fallback"` with `score=0.0`,
  while entity artifacts missing => deterministic entity extraction with `entities[].source="pattern"`.
- **FR-007**: System MUST allow operators to enable/disable the new intent and entity extraction engines independently via configuration.
- **FR-008**: System MUST not materially degrade user-perceived responsiveness for typical chat messages. Under a representative “normal load” for `/chat/advanced`, P95 end-to-end latency MUST NOT exceed 500 ms and the error rate attributable to the new engines MUST NOT increase by more than 0.5 percentage points versus the current baseline (measured with the evaluation tooling from Success Criteria SC-001/SC-004).
- **FR-009**: System MUST provide sufficient logging to diagnose model load failures and inference errors without exposing sensitive data.

### Key Entities *(include if feature involves data)*

- **Intent label**: A category representing what the user is trying to do (e.g., ask admission score, tuition, major info).
- **Confidence score**: A numeric indicator of certainty used for fallback behavior.
- **Extracted entity**: A structured item representing information found in the message with:
  - `label`: string
  - `text`: string (the extracted surface text)
  - `start`: number (start Unicode code point index, inclusive, in the original input)
  - `end`: number (end Unicode code point index, exclusive, in the original input)
  - `source`: string (e.g., `"model"` for model-based extraction, `"pattern"` or `"dictionary"` for deterministic rules)
- **Engine selection config**: Operator-controlled settings that select which intent/entity extraction approach is active.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a held-out evaluation set, intent classification improves versus the current baseline (higher accuracy/F1 or fewer wrong-intent cases), documented in a report.
- **SC-002**: The percentage of “wrong intent” outcomes decreases without an unacceptable increase in fallback rate (confidence threshold `T` and ambiguity margin `M` set via config; defaults calibrated and justified via offline evaluation on a held-out set).
- **SC-003**: For a predefined “golden set” of messages, outputs remain schema-compatible and stable across runs.
- **SC-004**: The system remains responsive for typical single-message requests under normal load, with no increase in error rate attributable to the new engines.

## Assumptions

- Users primarily chat in Vietnamese and ask about HUCE admissions information.
- Existing intent labels and entity labels remain valid; label renaming is out of scope unless required for correctness.
- No UI changes are required; this work targets backend NLP behavior only.
- If improved NLP components require external artifacts, they are provisioned in a way that does not break offline/local usage.
 - Session lifecycle (creation/reset) and storage of conversation history are owned by the service/API layer; this feature extends the NLP components so they can safely consume context derived from the current session while preserving the existing single-message NLP API contract.

## Clarifications

### Session 2026-04-02
- Q: When intent is ambiguous (top two intents are very close), how should fallback be triggered based on `score`? → A: `score` is the top intent probability; return fallback if `score < T` OR `(top1_prob - top2_prob) < M`.
- Q: What should the public `entities[]` item shape include? → A: `{ label, text, start, end, source }` where `start/end` are Unicode code point indices (start inclusive, end exclusive) in the original input, and `source` identifies the extraction engine.
- Q: How should thresholds `T` (confidence) and `M` (ambiguity margin) be set? → A: `T` and `M` are set via config, with default values chosen by offline calibration on the eval set (operator override allowed).
- Q: What happens when runtime artifacts required for improved NLP are missing/corrupted? → A: Component-wise degradation: if the intent model artifacts are missing/corrupted, return fallback intent with `score=0.0`; if the entity model artifacts are missing/corrupted, fall back to deterministic entity extraction (`entities[].source="pattern"`).
- Q: For `entities[].start/end`, should indices be in Unicode code points vs UTF-16 code units vs bytes? → A: Unicode code point indices (start inclusive, end exclusive).
### Session 2026-04-03
- Q: Is multi-turn context management (conversation history) in scope for this feature? → A: Yes — this feature includes improving context management so the chatbot can use conversation history within the current session to interpret the current message and answer more fluently, while preserving the existing NLP API contract and schema.
- Q: How much history should the context management consider by default? → A: Use the entire current session as context (until the user explicitly resets the session), relying on the existing “reset session” feature to clear history when needed.
