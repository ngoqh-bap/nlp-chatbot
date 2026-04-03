# Intent + NER (Model-based) + Efficient Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TF‑IDF/cosine intent + `underthesea` tokenization/NER with transformer-based engines, and upgrade context processing to be multi-turn aware while bounded and contract-safe.

**Architecture:** Introduce pluggable intent/entity engines (legacy vs transformer) behind stable interfaces; add a deterministic ContextProcessor that shapes session context into bounded model input for intent only; keep entity spans computed on the raw message to satisfy the contract.

**Tech Stack:** Python 3.13+, FastAPI (existing), pydantic v2 (existing), `transformers` + `torch` (model inference), pytest, ruff.

---

## Inputs / References (read before coding)

- Design doc: `docs/superpowers/specs/2026-04-03-intent-ner-models-v2-design.md`
- Feature spec: `specs/001-intent-ner-models/spec.md`
- Contract: `specs/001-intent-ner-models/contracts/nlp-pipeline-contract.md`
- Current implementations:
  - `nlu/pipeline.py`
  - `nlu/intent.py`
  - `nlu/entities.py`
  - `nlu/preprocess.py`
  - `services/nlp_service.py`
  - `tests/unit/test_context.py`, `tests/unit/test_intent.py`, `tests/unit/test_entities.py`, `tests/unit/test_preprocess.py`

## Scope check

This plan covers a single coherent subsystem: **NLP pipeline** (preprocessing, intent, entities) + **context shaping** for multi-turn. No UI changes.

## File structure decisions (locked)

**Modify (existing):**
- `nlu/pipeline.py`: orchestrate context shaping + engine selection + budgets; preserve output schema.
- `nlu/intent.py`: keep legacy TF‑IDF engine, add model engine + interface(s).
- `nlu/entities.py`: keep deterministic extractor, add model NER engine, guarantee `{label,text,start,end,source}`.
- `nlu/preprocess.py`: remove hard dependency on `underthesea`; keep normalization utilities; add deterministic tokenization fallback if needed.
- `services/nlp_service.py`: integrate session context into pipeline calls (while preserving the external API behavior).
- `config.py`: add engine selection + thresholds + budgets + context limits (aligned with existing config style).
- `pyproject.toml` (or equivalent): add `transformers`, `torch` if not present; remove `underthesea` only once fully unused.

**Create (new):**
- `nlu/context.py`: `ContextProcessor` + deterministic two-tier logic (structured memory + bounded text window + deterministic `context_summary` overflow handling).
- `nlu/engines/base.py`: engine protocols/interfaces + shared types (small, focused).
- `nlu/engines/intent_transformer.py`: transformer intent engine (load, infer, fallback math, budgets).
- `nlu/engines/ner_transformer.py`: transformer NER engine (offset mapping → spans).
- `nlu/engines/utils.py`: time-budget utilities + safe model loading helpers.
- `tests/unit/test_intent_transformer_engine.py`: deterministic tests with mocking.
- `tests/unit/test_ner_transformer_engine.py`: deterministic tests with mocking + offset mapping fixture.
- `tests/unit/test_context_processor.py`: bounded context + summary determinism.

> Note: If the repo prefers fewer files, these can be collapsed, but keep responsibilities separate: context shaping, intent inference, NER inference, pipeline orchestration.

## Global invariants (must hold after every task)

- Pipeline output schema stays exactly: `{"intent": str, "score": number, "entities": list}`.
- `score` is always in `[0.0, 1.0]`.
- `score` MUST represent the **top-1 intent probability** (`top1`), and `top2` MUST be the second-highest probability.
  - Transformer intent uses `softmax(logits)` probabilities.
  - Legacy intent uses probabilities derived from per-intent cosine similarities via `softmax(similarity / temperature)`.
- `top2` is an internal value used only to evaluate the ambiguity margin \(M\); it is **not** returned in the public pipeline output.
- Intent fallback triggers when `top1 < T` OR `(top1 - top2) < M` (and on errors/timeouts with `score=0.0`).
- Entity spans `start/end` are Unicode code point indices into the **raw message** passed to `NLPPipeline.analyze(text)`.
- No raw user text in production logs.
- Tests do **not** require downloading model weights; inference must be mockable.

---

### Task 0: Fix baseline legacy intent wiring + make legacy score contract-compliant (probability)

**Why first:** The repo currently instantiates `IntentDetector` with the wrong constructor args; also the contract requires `score` be a top-1 probability, so the legacy engine must output probabilities (not raw cosine similarity).

**Files:**
- Modify: `nlu/intent.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_intent.py`

- [ ] **Step 1: Write failing test reproducing current wiring**

Add to `tests/unit/test_intent.py`:

```python
def test_pipeline_constructs_intent_detector_without_type_error():
    from nlu.pipeline import NLPPipeline
    NLPPipeline()  # should not raise due to ctor mismatch
```

- [ ] **Step 2: Run test to verify failure**

Run (PowerShell): `pwsh -NoProfile -Command "pytest tests/unit/test_intent.py -k pipeline_constructs_intent_detector -v"`  
Expected: FAIL with `TypeError` (constructor arg mismatch)

- [ ] **Step 3: Fix the constructor mismatch**

Pick one consistent approach:
- **Preferred:** simplify `IntentDetector.__init__` to only require `intent_samples` and `threshold` (remove `intent_keyword_backoff` if it’s unused), OR
- Pass a real (possibly empty) `intent_keyword_backoff` from `NLPPipeline`.

Make sure `NLPPipeline.__init__` and `IntentDetector.__init__` match.

- [ ] **Step 4: Make legacy outputs probability-like**

Update legacy intent scoring to a probability distribution so the contract holds in legacy mode:
- Compute cosine similarity per intent
- Convert similarities to probabilities via `softmax(similarity / temperature)`
- Return:
  - `score = top1_prob`
  - `top2` for margin checks
  - intent selection with the same `T/M` fallback logic as the model engine

This preserves rollback capability while keeping the public contract semantics consistent.

- [ ] **Step 5: Run the test**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_intent.py -k pipeline_constructs_intent_detector -v"`  
Expected: PASS

- [ ] **Step 6: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/intent.py nlu/pipeline.py tests/unit/test_intent.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"fix: align legacy intent detector wiring and probability scoring\""`

### Task 1: Add configuration knobs (engines, thresholds, budgets, context limits)

**Files:**
- Modify: `config.py`
- Modify: `nlu/pipeline.py` (only to read config defaults if needed)
- Test: `tests/unit/test_exceptions.py` (only if config validation errors already covered) or new `tests/unit/test_config_nlu.py`

- [ ] **Step 1: Write failing tests for config defaults**

Create `tests/unit/test_config_nlu.py`:

```python
def test_default_intent_engine_is_legacy():
    from config import get_nlu_intent_engine
    assert get_nlu_intent_engine() == "legacy"

def test_default_entity_engine_is_deterministic():
    from config import get_nlu_entity_engine
    assert get_nlu_entity_engine() == "deterministic"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_config_nlu.py -v"`  
Expected: FAIL (missing getters)

- [ ] **Step 3: Implement config getters (minimal)**

In `config.py`, add getters (names can be adjusted to match existing style):
- `get_nlu_intent_engine() -> str`
- `get_nlu_entity_engine() -> str`
- `get_intent_threshold_T() -> float`
- `get_intent_margin_M() -> float`
- `get_intent_budget_ms() -> int`
- `get_entity_budget_ms() -> int`
- `get_context_history_limit() -> int` (already exists)
- `get_context_max_chars() -> int`
- `get_context_turns_for_model() -> int`

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_config_nlu.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add config.py tests/unit/test_config_nlu.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: add NLU engine and budget config\""`

---

### Task 2: Implement deterministic `ContextProcessor` (two-tier context shaping)

**Files:**
- Create: `nlu/context.py`
- Modify: `services/nlp_service.py` (later tasks will wire it fully)
- Test: `tests/unit/test_context_processor.py`

- [ ] **Step 1: Write failing tests for bounded model input + determinism + summary support**

Create `tests/unit/test_context_processor.py`:

```python
def test_model_input_is_bounded_by_max_chars():
    from nlu.context import ContextProcessor
    cp = ContextProcessor(max_chars=50, turns_for_model=3)
    ctx = {"conversation_history": [{"message": "A" * 100}], "context_summary": "S" * 100}
    out = cp.build_intent_input("hi", ctx)
    assert len(out) <= 50

def test_history_retention_is_not_modified_by_processor():
    from nlu.context import ContextProcessor
    cp = ContextProcessor(max_chars=200, turns_for_model=3)
    hist = [{"message": "m1"}, {"message": "m2"}]
    ctx = {"conversation_history": hist.copy()}
    _ = cp.build_intent_input("m3", ctx)
    assert ctx["conversation_history"] == hist

def test_overflow_updates_summary_deterministically():
    from nlu.context import ContextProcessor
    cp = ContextProcessor(max_chars=40, turns_for_model=10)
    ctx = {"conversation_history": [{"message": "m1 " * 50}, {"message": "m2"}], "context_summary": ""}
    out1, ctx1 = cp.build_intent_input_and_update_context("hi", ctx)
    out2, ctx2 = cp.build_intent_input_and_update_context("hi", ctx)
    assert out1 == out2
    assert ctx1["context_summary"] == ctx2["context_summary"]
    assert len(out1) <= 40
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_context_processor.py -v"`  
Expected: FAIL (module/class missing)

- [ ] **Step 3: Implement `ContextProcessor` with deterministic two-tier behavior**

In `nlu/context.py`, implement:
- `ContextProcessor.__init__(max_chars: int, turns_for_model: int)`
- `build_intent_input(message: str, current_context: dict) -> str`
- `build_intent_input_and_update_context(message: str, current_context: dict) -> tuple[str, dict]`

Deterministic behavior:
- Build a string from up to last `turns_for_model` history entries (if present)
- Prepend `context_summary` if present (may be empty)
- Append `User: {message}`
- If over `max_chars`, deterministically move the oldest textual turns into `context_summary` (e.g., append a truncated copy of dropped turn messages), then truncate from the left as a final safety net while preserving the end of the string (contains the user message)

> Keep it simple and deterministic; no summarization model.

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_context_processor.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/context.py tests/unit/test_context_processor.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: add deterministic ContextProcessor for intent\""`

---

### Task 3: Refactor intent into engine interface + keep legacy TF‑IDF engine

**Files:**
- Create: `nlu/engines/base.py`
- Modify: `nlu/intent.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_intent.py`

- [ ] **Step 1: Add a failing test for engine selection returning legacy behavior**

Extend `tests/unit/test_intent.py` (or add a focused new test):

```python
def test_legacy_intent_engine_still_available():
    from nlu.intent import LegacyTfidfIntentEngine
    engine = LegacyTfidfIntentEngine(intent_samples={"x": [["a"]]}, threshold=0.0)
    intent, score = engine.detect("a")
    assert intent == "x"
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_intent.py -v"`  
Expected: FAIL (class not present / API mismatch)

- [ ] **Step 3: Implement `IntentEngine` protocol + wrap legacy**

In `nlu/engines/base.py`:
- Define `IntentResult = tuple[str, float]`
- Define protocol/interface `IntentEngine` with `detect(text: str) -> IntentResult`

In `nlu/intent.py`:
- Keep existing TF‑IDF code, but introduce `LegacyTfidfIntentEngine` that adapts current `IntentDetector` behavior to `detect(text)`.
- Ensure legacy `score` remains contract-compliant as **top-1 probability** (per Task 0): convert cosine similarities to probabilities via softmax, then apply `T/M` fallback and return `score=top1_prob`.

In `nlu/pipeline.py`:
- Replace direct `IntentDetector` usage with engine instance selection.

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_intent.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/engines/base.py nlu/intent.py nlu/pipeline.py tests/unit/test_intent.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"refactor: wrap legacy TF-IDF intent as engine\""`

---

### Task 4: Add transformer intent engine (mockable) with `T`/`M` fallback + budget

**Files:**
- Create: `nlu/engines/intent_transformer.py`
- Create: `nlu/engines/utils.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_intent_transformer_engine.py`

- [ ] **Step 1: Write failing tests for fallback math and error/timeout behavior**

Create `tests/unit/test_intent_transformer_engine.py`:

```python
def test_fallback_triggers_on_low_top1():
    from nlu.engines.intent_transformer import decide_intent
    intent, score = decide_intent(
        labels=["a", "b"],
        probs=[0.2, 0.8],
        t=0.9,
        m=0.0,
    )
    assert intent == "fallback"
    assert score == 0.8

def test_fallback_triggers_on_small_margin():
    from nlu.engines.intent_transformer import decide_intent
    intent, score = decide_intent(
        labels=["a", "b"],
        probs=[0.51, 0.49],
        t=0.0,
        m=0.05,
    )
    assert intent == "fallback"
    assert score == 0.51

def test_single_label_has_top2_zero():
    from nlu.engines.intent_transformer import decide_intent
    intent, score = decide_intent(labels=["a"], probs=[1.0], t=0.0, m=0.5)
    assert intent == "a"
    assert score == 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_intent_transformer_engine.py -v"`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `decide_intent()` pure function + engine skeleton**

In `nlu/engines/intent_transformer.py`:
- Implement `decide_intent(labels: list[str], probs: list[float], t: float, m: float) -> tuple[str, float]`
  - pick top1/top2
  - return `(label, top1)` or `("fallback", top1)` based on rules
  - validate/clamp probs to `[0,1]` defensively
- Implement `TransformerIntentEngine` class that:
  - loads artifacts if present (but in tests allow dependency injection/mocking)
  - enforces budget via a wrapper in `nlu/engines/utils.py`
  - on any load/infer failure: returns `("fallback", 0.0)`

In `nlu/pipeline.py`:
- Wire engine selection from config:
  - if `NLU_INTENT_ENGINE=transformer` → use `TransformerIntentEngine`
  - else → legacy

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_intent_transformer_engine.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/engines/intent_transformer.py nlu/engines/utils.py nlu/pipeline.py tests/unit/test_intent_transformer_engine.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: add transformer intent engine with T/M fallback\""`

---

### Task 5: Refactor entities into engine interface + guarantee `{label,text,start,end,source}` for deterministic extraction

**Files:**
- Modify: `nlu/entities.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_entities.py`

- [ ] **Step 1: Add failing test asserting deterministic entities include spans**

Extend `tests/unit/test_entities.py`:

```python
def test_deterministic_entities_have_spans():
    from nlu.entities import EntityExtractor
    ex = EntityExtractor(data_dir="data", patterns_path="data/entity.json", synonym_map={})
    ents = ex.extract("điểm chuẩn")
    for e in ents:
        assert set(e.keys()) >= {"label", "text", "start", "end", "source"}
        assert isinstance(e["start"], int) and isinstance(e["end"], int)
        assert 0 <= e["start"] <= e["end"] <= len("điểm chuẩn")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_entities.py -v"`  
Expected: FAIL (start/end missing)

- [ ] **Step 3: Implement span computation for deterministic entities**

In `nlu/entities.py`:
- When adding pattern/dictionary entities, compute `start/end` by searching in the **raw input**:
  - Use `normalize_text` only for matching decisions, but compute spans against the raw string by finding best-effort substring match of the extracted surface form (case-insensitive recommended, but deterministic).
  - If a span can’t be found reliably, omit the entity (or set `start/end` to a safe value only if contract allows; contract expects spans, so prefer dropping).
- Ensure dedup keeps `start/end` and `source`.

In `nlu/pipeline.py`:
- Ensure entity engine always returns list of full-shape entities.

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_entities.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/entities.py nlu/pipeline.py tests/unit/test_entities.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"fix: include spans in deterministic entity extraction\""`

---

### Task 6: Add transformer NER engine (raw-message only) using offset mappings

**Files:**
- Create: `nlu/engines/ner_transformer.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_ner_transformer_engine.py`

- [ ] **Step 1: Write failing tests for BIO-to-spans reconstruction using offsets**

Create `tests/unit/test_ner_transformer_engine.py` that tests the span reconstruction helper as a pure function:

```python
def test_offsets_to_spans_merges_bio_tokens():
    from nlu.engines.ner_transformer import offsets_to_spans
    text = "ngành kiến trúc"
    offsets = [(0, 5), (6, 10), (11, 15)]  # ngành | kiến | trúc
    tags = ["O", "B-TEN_NGANH", "I-TEN_NGANH"]
    spans = offsets_to_spans(text=text, offsets=offsets, tags=tags)
    assert spans == [
        {"label": "TEN_NGANH", "text": "kiến trúc", "start": 6, "end": 15, "source": "model"}
    ]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_ner_transformer_engine.py -v"`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `offsets_to_spans()` + engine skeleton**

In `nlu/engines/ner_transformer.py`:
- Implement pure function `offsets_to_spans(text, offsets, tags)`:
  - Merge consecutive `B-XXX` + `I-XXX` into a single entity
  - Ignore `O`
  - Use offsets directly; produce `start/end` as given (Python indices)
  - Set `source="model"`
- Implement `TransformerNerEngine.extract(text: str) -> list[dict]`:
  - Runs on raw `text` only
  - Uses tokenizer with `return_offsets_mapping=True`
  - Require a fast tokenizer (e.g., `AutoTokenizer(..., use_fast=True)`); if offsets are unavailable, **do not emit model entities** and fall back to deterministic-only to avoid incorrect spans
  - Enforces budget; on failure returns `[]` (pipeline will merge with deterministic if configured)

- [ ] **Step 4: Wire into pipeline with engine toggle**

In `nlu/pipeline.py`:
- If `NLU_ENTITY_ENGINE=transformer`, run deterministic + model, then merge/dedup.
- If `NLU_ENTITY_ENGINE=deterministic`, run deterministic only.
- If `NLU_ENTITY_ENGINE=off`, return `entities=[]`.

- [ ] **Step 5: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_ner_transformer_engine.py -v"`  
Expected: PASS

- [ ] **Step 6: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/engines/ner_transformer.py nlu/pipeline.py tests/unit/test_ner_transformer_engine.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: add transformer NER engine with span reconstruction\""`

---

### Task 7: Remove `underthesea` usage from preprocessing (keep deterministic fallback)

**Files:**
- Modify: `nlu/preprocess.py`
- Modify: `pyproject.toml` (later, after no remaining imports)
- Test: `tests/unit/test_preprocess.py`

- [ ] **Step 1: Add failing test asserting preprocess works without `underthesea` installed**

In `tests/unit/test_preprocess.py`, add a test that imports and tokenizes without requiring underthesea. (Implementation detail: the current code already has an ImportError fallback; this task ensures we remove dependency, not just hide it.)

```python
def test_tokenize_is_pure_python():
    from nlu.preprocess import tokenize_and_map
    toks = tokenize_and_map("Xin chào", {})
    assert isinstance(toks, list)
```

- [ ] **Step 2: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_preprocess.py -v"`  
Expected: PASS or FAIL depending on current expectations

- [ ] **Step 3: Implement: delete `underthesea` import and rely on deterministic split**

In `nlu/preprocess.py`:
- Remove the `try: from underthesea import word_tokenize` block
- Ensure tokenization is deterministic (e.g., `norm.split()`)
- Keep `normalize_text` as-is

- [ ] **Step 4: Run preprocess + full unit suite**

Run: `pwsh -NoProfile -Command "pytest tests/unit -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/preprocess.py tests/unit/test_preprocess.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"refactor: remove underthesea from preprocessing\""`

---

### Task 8: Wire context-aware intent into `NLPService` + pipeline, preserve current behavior

**Files:**
- Modify: `services/nlp_service.py`
- Modify: `nlu/pipeline.py`
- Test: `tests/unit/test_context.py`, `tests/integration/test_api.py`

- [ ] **Step 1: Write failing test that intent engine sees context-shaped input**

Add to `tests/unit/test_context.py` (or new unit test with a fake intent engine):

```python
def test_intent_uses_context_processor(monkeypatch):
    from nlu.pipeline import NLPPipeline
    p = NLPPipeline()
    seen = {}

    class FakeEngine:
        def detect(self, text: str):
            seen["text"] = text
            return ("fallback", 0.0)

    monkeypatch.setattr(p, "_intent_engine", FakeEngine())
    p.analyze("hi", context={"conversation_history": [{"message": "prev"}]})
    assert "prev" in seen["text"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_context.py -v"`  
Expected: FAIL (`analyze` doesn’t accept context yet)

- [ ] **Step 3: Implement internal context-aware analyze without breaking contract**

In `nlu/pipeline.py`:
- Keep the public `analyze(text: str)` signature if it’s used elsewhere, but add:
  - `analyze_with_context(text: str, context: dict) -> dict`
  - Or accept optional `context: dict | None = None` while still supporting existing call sites.
- Use `ContextProcessor` only for **intent input**.
- Keep entity extraction on raw `text` only.

In `services/nlp_service.py`:
- Pass `current_context` into the pipeline call when analyzing messages.

- [ ] **Step 4: Run unit + integration tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_context.py -v"`  
Expected: PASS  

Run: `pwsh -NoProfile -Command "pytest tests/integration/test_api.py -v"`  
Expected: PASS (schema unchanged)

- [ ] **Step 5: Clarify service-layer fallback labeling**

`services/nlp_service.py` currently mutates `analysis["intent"]` to `"fallback_response"`. Required behavior after this step:
- `analysis["intent"]` MUST remain the pipeline contract value (`"fallback"` or a canonical intent label) and MUST NOT be rewritten to `"fallback_response"`.
- If routing needs a separate marker, store it in the response payload (e.g., `response["type"] = "fallback_response"`) without mutating the NLP analysis contract fields.

Add/adjust an integration test in `tests/integration/test_api.py` to lock this behavior.

- [ ] **Step 6: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/pipeline.py services/nlp_service.py tests/unit/test_context.py tests/integration/test_api.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: use session context for intent analysis\""`

---

### Task 9: Add budgets + logging (no raw text) for model engines

**Files:**
- Modify: `nlu/engines/utils.py`
- Modify: `nlu/engines/intent_transformer.py`
- Modify: `nlu/engines/ner_transformer.py`
- Test: `tests/unit/test_sanitize.py` (if logging sanitizer exists) or new tests for “does not log raw text” where feasible

- [ ] **Step 1: Add tests for budget wrapper behavior**

```python
def test_budget_timeout_returns_fallback():
    from nlu.engines.utils import run_with_budget_ms
    def slow():
        raise TimeoutError()
    ok, val = run_with_budget_ms(1, slow, fallback=("fallback", 0.0))
    assert val == ("fallback", 0.0)
```

- [ ] **Step 2: Implement `run_with_budget_ms()` helper**

Implement a deterministic budget mechanism (prefer cooperative/time checks rather than threads if possible; if threads are used, keep it minimal and testable).

- [ ] **Step 3: Add structured logging**

In engines:
- Log only:
  - request id (MUST be propagated from middleware; if an engine cannot access it directly, log MUST occur at the service/pipeline boundary where request id is available)
  - component + engine name + elapsed ms + fallback reason
- Do not log raw `text` or full context strings.

- [ ] **Step 4: Run unit tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/engines/utils.py nlu/engines/intent_transformer.py nlu/engines/ner_transformer.py tests/unit"`  
Run: `pwsh -NoProfile -Command "git commit -m \"feat: enforce inference budgets and structured logging\""`

---

### Task 10: Dependency cleanup (`underthesea` removal) + docs update

**Files:**
- Modify: `pyproject.toml`
- Modify: `pytest.ini` (remove underthesea warning ignore when safe)
- Modify: `README.md` (remove underthesea mention, update NLP description)
- Test: full suite

- [ ] **Step 1: Verify no imports remain**

Run: `pwsh -NoProfile -Command "python -c 'import nlu.preprocess, nlu.entities, nlu.pipeline'"`  
Expected: success

- [ ] **Step 2: Remove dependency + warning ignore**

Update `pyproject.toml` to remove `underthesea==...` only after code no longer imports it anywhere.  
Update `pytest.ini` to remove underthesea-specific ignores if present.

- [ ] **Step 3: Update README**

Replace legacy bullets with transformer engines and deterministic fallback.

- [ ] **Step 4: Run full verification**

Run: `pwsh -NoProfile -Command "pytest -v"`  
Expected: PASS

Run: `pwsh -NoProfile -Command "ruff check ."`  
Expected: no new errors

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add pyproject.toml pytest.ini README.md"`  
Run: `pwsh -NoProfile -Command "git commit -m \"chore: remove underthesea dependency after migration\""`

---

## Execution notes (important)

- **No weight downloads in tests**: transformer engines must be constructed so tests can stub model outputs (e.g., dependency injection for logits/tags).
- **GPU optional**: runtime can use GPU if available, but correctness must not depend on GPU.
- **Entity offsets**: always computed relative to raw user message; do not run NER on context-prefixed strings.

---

### Task 11: Lock spec edge cases with tests (empty/symbols-only + extremely long)

**Files:**
- Create: `tests/unit/test_pipeline_edge_cases.py`
- Modify: `nlu/pipeline.py` (if needed)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pipeline_edge_cases.py`:

```python
def test_empty_input_returns_fallback_and_no_entities():
    from nlu.pipeline import NLPPipeline
    p = NLPPipeline()
    out = p.analyze("")
    assert out["intent"] == "fallback"
    assert out["score"] == 0.0
    assert out["entities"] == []

def test_symbols_only_returns_fallback_and_no_entities():
    from nlu.pipeline import NLPPipeline
    p = NLPPipeline()
    out = p.analyze("!!!")
    assert out["intent"] == "fallback"
    assert out["score"] == 0.0
    assert out["entities"] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_pipeline_edge_cases.py -v"`  
Expected: FAIL until pipeline implements explicit handling

- [ ] **Step 3: Implement minimal behavior**

In `nlu/pipeline.py`, before any engine calls:
- If normalized text is empty: return fallback with score `0.0` and `entities=[]`.

- [ ] **Step 4: Run tests**

Run: `pwsh -NoProfile -Command "pytest tests/unit/test_pipeline_edge_cases.py -v"`  
Expected: PASS

- [ ] **Step 5: Commit**

Run: `pwsh -NoProfile -Command "git add nlu/pipeline.py tests/unit/test_pipeline_edge_cases.py"`  
Run: `pwsh -NoProfile -Command "git commit -m \"test: lock NLP pipeline edge cases\""`

