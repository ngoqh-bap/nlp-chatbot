# Intent + NER (PhoBERT, Approach A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use @superpowers:subagent-driven-development (recommended) or @superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship **PhoBERT-based** transformer intent + optional **PhoBERT-family token-classification NER** (community Vietnamese NER checkpoint **or** operator fine-tune on a local BIO label set) behind config flags, with **GPU-first** inference defaults (RTX 5060–class), **local artifact dirs**, and contract-safe degradation—without breaking the stable `NLPPipeline` output schema.

**Architecture:** Keep **Approach A** from `docs/superpowers/specs/2026-04-03-intent-ner-models-v2-design.md`: separate **sequence-classification** intent model and separate **`AutoModelForTokenClassification`** NER head on a **PhoBERT** backbone; **deterministic** preprocessing; **NER only on the raw user message** (`TransformerNerEngine`); intent may use **context-shaped** text via `ContextProcessor`. Deterministic pattern/dictionary extraction stays on and **merges** with neural spans. Legacy TF‑IDF remains the default intent engine until operators pin artifacts and enable transformers.

**Tech Stack:** Python 3.13+, FastAPI, pydantic v2, `transformers` + `torch`, pytest, ruff. Approved encoders: **PhoBERT** (`vinai/phobert-base` or `vinai/phobert-large-v2` for training/bake-off; runtime loads **fine-tuned local dirs**).

**Context:** Prefer implementing in a **dedicated git worktree** (see @superpowers:using-git-worktrees) so `main` stays mergeable.

---

## Inputs / references (read before coding)

| Document | Path |
|----------|------|
| Approved design (model stack, RTX 5060 notes, offsets) | `docs/superpowers/specs/2026-04-03-intent-ner-models-v2-design.md` |
| Feature spec + acceptance | `specs/001-intent-ner-models/spec.md` |
| Pipeline contract | `specs/001-intent-ner-models/contracts/nlp-pipeline-contract.md` |
| Current NLP code | `nlu/pipeline.py`, `nlu/intent.py`, `nlu/entities.py`, `nlu/preprocess.py`, `nlu/context.py` |
| Config | `config.py`, `env.example` |

## Scope check

Single subsystem: **NLU engines + pipeline wiring + config + tests**. Optional follow-ups: **PhoBERT intent + NER training scripts**, **sticky structured memory** (`sticky_entities`), README/ops docs. No Reflex/UI changes in core tasks.

## Repo snapshot — baseline already landed (verify; do not redo blindly)

The following are **expected on `main`** (run `pytest`; only fix if regressions appear):

| Area | Status | Pointers |
|------|--------|----------|
| Legacy intent | TF‑IDF centroids + **softmax** probs + **T/M** fallback | `nlu/intent.py` (`IntentDetector`), `LegacyTfidfIntentEngine` |
| Transformer intent | Local `AutoModelForSequenceClassification` + budgets | `nlu/engines/intent_transformer.py` |
| Transformer NER | Local `AutoModelForTokenClassification` + BIO → spans, raw `text` only | `nlu/engines/ner_transformer.py` |
| Pipeline wiring | `NLU_INTENT_ENGINE` / `NLU_ENTITY_ENGINE`, merge neural + deterministic | `nlu/pipeline.py` |
| Context shaping | `ContextProcessor` bounded + deterministic overflow | `nlu/context.py`, `tests/unit/test_context_processor.py` |
| NLU config | paths, device, autocast, budgets | `config.py`, `tests/unit/test_config_nlu.py` |
| Intent training | `scripts/train_intent_phobert.py` | CLI fine-tune on `data/intent.csv` |

**Remaining gap (NER, design-aligned):** Runtime code expects a **Hugging Face save dir** at `NLU_NER_MODEL_DIR` with `config.json`, tokenizer, weights, and **`id2label` BIO tags**. The repo **does not** ship that checkpoint. Operators must either (**A**) download a **community PhoBERT-based Vietnamese NER** `TokenClassification` model from the Hub and save it locally, or (**B**) **fine-tune PhoBERT** on a project-specific BIO corpus and point the env var at the export dir—see §**PhoBERT-based NER artifacts** below. Optional: add `scripts/train_ner_phobert.py` mirroring the intent script.

## File structure (this effort)

**Expected on `main` (verify; add only if missing):**

| File | Responsibility |
|------|----------------|
| `nlu/engines/__init__.py` | Re-export public engine classes |
| `nlu/engines/base.py` | `Protocol`s: `IntentEngine`, `NerEngine` |
| `nlu/engines/utils.py` | `run_with_budget_ms`, device/autocast helpers |
| `nlu/engines/intent_transformer.py` | PhoBERT **sequence classification** from local dir; `decide_intent(T, M)` |
| `nlu/engines/ner_transformer.py` | PhoBERT-family **`AutoModelForTokenClassification`** from local dir; BIO → spans; **raw message** only |
| `tests/unit/test_intent_transformer_engine.py` | `decide_intent` + mocks |
| `tests/unit/test_ner_transformer_engine.py` | `offsets_to_spans` + mocks |
| `tests/unit/test_pipeline_engines.py` | Engine selection + independence |
| `tests/unit/test_engine_budget.py` | Budget helper |
| `tests/unit/test_pipeline_long_input.py` | Long-input semantics |

**Optional later:**

| File | Responsibility |
|------|----------------|
| `scripts/train_intent_phobert.py` | Fine-tune PhoBERT on `data/intent.csv` → artifact dir (may already exist) |
| `scripts/train_ner_phobert.py` | Fine-tune **PhoBERT** `AutoModelForTokenClassification` on operator BIO data → artifact dir (optional; see Task 11) |

**Modify:**

| File | Change |
|------|--------|
| `nlu/pipeline.py` | Construct legacy vs transformer engines from config; route intent/entity budgets; merge NER with deterministic entities |
| `nlu/intent.py` | Optional thin `LegacyTfidfIntentEngine` adapter class wrapping `IntentDetector` (keeps `detect` API stable) |
| `nlu/entities.py` | Optional `DeterministicEntityEngine` facade if it clarifies merge; else keep `EntityExtractor` |
| `config.py` | `get_intent_model_path()`, `get_ner_model_path()`, `get_nlu_device()`, `get_nlu_use_autocast()` (names may match `env.example`) |
| `env.example` | Document new vars |
| `main.py` or middleware | Pass **request id** into NLP path if not already available for structured logs |

## Global invariants (must hold after every task)

- Output of `NLPPipeline.analyze` / `analyze_with_context`: `{"intent": str, "score": float, "entities": list}` unchanged.
- `score` ∈ `[0.0, 1.0]` and is **top-1** probability (legacy softmax or transformer softmax).
- Fallback: `top1 < T` OR `(top1 - top2) < M` → `intent="fallback"` with `score=top1` (except artifact/inference failure → `score=0.0` per contract).
- Entity `start`/`end`: Unicode code point indices into **raw** `text` argument to `analyze_with_context`.
- **Extremely long input** (spec + design): intent inference (legacy or transformer) uses **bounded** text (context processor + optional `NLU_INTENT_MAX_CHARS` truncation of `intent_input` only). **Deterministic entities** always run on the **full** original `text`. If transformer intent completes on truncated input, `score` is still `top1` from that forward pass, including when the resolved intent is `"fallback"`. If intent inference **fails or times out**, `intent="fallback"`, `score=0.0`.
- **No** model weight download in CI/tests; inject mocks or tiny random modules.
- **No** raw user message in production logs.

### Inference budget contract (`nlu/engines/utils.py`)

Implement **`run_with_budget_ms(budget_ms: int, fn: Callable[[], T], *, on_timeout: T) -> T`** (add `Callable` import from `typing.collections.abc` or `typing`) with behavior locked by tests:

| Mechanism | Detail |
|-----------|--------|
| **Primary** | Run `fn` in a **worker thread** via `concurrent.futures.ThreadPoolExecutor`; `future.result(timeout=budget_ms/1000.0)`. On `concurrent.futures.TimeoutError`, return `on_timeout` (e.g. `("fallback", 0.0)` for intent, `[]` for NER). |
| **CUDA caveat** | GPU kernels may not stop when the thread times out; document that this is **best-effort** wall-clock bounding for UX, not a hard CUDA cancel. Prefer `timeout` + discard result + log `fallback_reason=timeout`. |
| **Tests** | Unit-test: `fn` sleeps longer than budget → returns `on_timeout`; fast `fn` → returns real result. No real `torch` required for these tests (use `time.sleep` stub). |

NER and intent **both** use this helper so “budget” is not vague.

---

## PhoBERT-based NER artifacts (design `2026-04-03-intent-ner-models-v2-design.md` §Model choices, §NER)

The approved **Entities (neural)** row: **token classification (BIO)** on a **PhoBERT backbone**, via **`AutoModelForTokenClassification` + fast `AutoTokenizer` + `return_offsets_mapping=True`**, running on the **raw current user message** only; **merge** with deterministic pattern/dictionary entities; span **`source="model"`**; **`start`/`end`** are Unicode code point indices into that raw message (see design §NER, §Offset contract).

**Two supported acquisition paths (operator chooses one per deployment):**

| Path | When | Actions |
|------|------|---------|
| **A — Community checkpoint** | Fastest path; public labels (e.g. PER/ORG/LOC) suffice | On [HuggingFace Hub](https://huggingface.co/models), find a **`TokenClassification`** model whose config shows a **PhoBERT** encoder (e.g. `vinai/phobert-base` in `config.json` / base model metadata). Run `snapshot_download` or `from_pretrained` + `save_pretrained(local_dir)`. Set **`NLU_NER_MODEL_DIR`** to that directory. Set **`NLU_ENTITY_ENGINE=transformer`**. |
| **B — Fine-tune on your label set** | Domain labels (e.g. admission-specific types) must match CSV/KB | Prepare a **word- or token-aligned BIO** dataset (same label vocabulary you want at runtime). Fine-tune **from `vinai/phobert-base`** (or `vinai/phobert-large-v2` per latency bake-off) with HF **`AutoModelForTokenClassification`**; export a normal **`save_pretrained` dir**. Point **`NLU_NER_MODEL_DIR`** there. Use **Task 11** script if implemented. |

**Label / merge rules (runtime):**

- `TransformerNerEngine` reads **`model.config.id2label`**; predictions must be **BIO** strings (with **`O`**) so `offsets_to_spans` in `nlu/engines/ner_transformer.py` can merge spans.
- Community models may use different type names than `entity.json`; **deterministic** extraction still fills gaps. Neural spans **dedupe** with deterministic on `(label, start, end)` in `NLPPipeline._merge_entity_lists`.

**Open design item (not blocking code):** Pin **exact Hub repo id** for Path A after a short eval on Vietnamese admission utterances (design “Exact checkpoint IDs”).

---

### Task 1: Engine protocols + legacy adapter + pipeline selection stub

**Files:**

- Create: `nlu/engines/base.py`
- Create: `nlu/engines/__init__.py`
- Modify: `nlu/intent.py` — add `LegacyTfidfIntentEngine`
- Modify: `nlu/pipeline.py` — select engine from `get_nlu_intent_engine()` (default `legacy`)

**Why:** Unblocks transformer modules without changing default behavior.

- [ ] **Step 1: Failing test — legacy path still default**

Create `tests/unit/test_pipeline_engines.py`:

```python
def test_default_intent_engine_is_legacy(monkeypatch):
    monkeypatch.delenv("NLU_INTENT_ENGINE", raising=False)
    from importlib import reload
    import config as cfg
    reload(cfg)
    from nlu.pipeline import NLPPipeline
    p = NLPPipeline()
    assert getattr(p, "_intent_engine_name", None) in (None, "legacy") or cfg.get_nlu_intent_engine() == "legacy"
```

Adjust assertion to match implementation: e.g. `p._intent_engine` is instance of `LegacyTfidfIntentEngine` or pipeline stores `engine_kind == "legacy"`.

- [ ] **Step 2: Run test — expect FAIL**

```powershell
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; pytest tests/unit/test_pipeline_engines.py -v"
```

Expected: FAIL (missing attributes / no selection).

- [ ] **Step 3: Implement**

In `nlu/engines/base.py`:

```python
from typing import Protocol, Tuple, List, Dict, Any, runtime_checkable

IntentResult = Tuple[str, float]


@runtime_checkable
class IntentEngine(Protocol):
    def detect(self, text: str, synonym_map: Dict[str, str], normalize) -> IntentResult: ...


@runtime_checkable
class NerEngine(Protocol):
    def extract(self, text: str) -> List[Dict[str, Any]]: ...
```

In `nlu/intent.py`, add:

```python
class LegacyTfidfIntentEngine:
    def __init__(self, detector: "IntentDetector") -> None:
        self._det = detector

    def detect(self, text: str, synonym_map: Dict[str, str], normalize) -> Tuple[str, float]:
        return self._det.detect(text, synonym_map, normalize)
```

In `nlu/pipeline.py`, replace direct `self._intent_detector.detect(...)` with `self._intent_engine.detect(...)` where `self._intent_engine = LegacyTfidfIntentEngine(self._intent_detector)` built when `get_nlu_intent_engine() == "legacy"`.

- [ ] **Step 4: Run test**

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; git add nlu/engines/base.py nlu/engines/__init__.py nlu/intent.py nlu/pipeline.py tests/unit/test_pipeline_engines.py; git commit -m \"feat(nlu): add IntentEngine protocol and legacy adapter\""
```

---

### Task 2: Pure `decide_intent` + unit tests (transformer math)

**Files:**

- Create: `nlu/engines/intent_transformer.py` (functions only first)
- Test: `tests/unit/test_intent_transformer_engine.py`

- [ ] **Step 1: Failing tests**

```python
def test_decide_intent_low_top1():
    from nlu.engines.intent_transformer import decide_intent
    intent, score = decide_intent(["a", "b"], [0.2, 0.8], t=0.9, m=0.0)
    assert intent == "fallback" and score == 0.2

def test_decide_intent_small_margin():
    from nlu.engines.intent_transformer import decide_intent
    intent, score = decide_intent(["a", "b"], [0.51, 0.49], t=0.0, m=0.05)
    assert intent == "fallback" and score == 0.51
```

- [ ] **Step 2: Run — FAIL**

```powershell
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; pytest tests/unit/test_intent_transformer_engine.py -v"
```

- [ ] **Step 3: Implement `decide_intent`**

Match `IntentDetector` semantics: when fallback, return `("fallback", top1)`.

```python
def decide_intent(labels: list[str], probs: list[float], t: float, m: float) -> tuple[str, float]:
    if not labels or not probs:
        return "fallback", 0.0
    ranked = sorted(zip(probs, labels), key=lambda x: -x[0])
    top1_p, top1_l = ranked[0]
    top2_p = ranked[1][0] if len(ranked) > 1 else 0.0
    if top1_p < t or (top1_p - top2_p) < m:
        return "fallback", top1_p
    return top1_l, top1_p
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `test(nlu): add decide_intent for transformer softmax`

---

### Task 3: Config — local artifact paths + device + autocast

**Files:**

- Modify: `config.py`
- Modify: `env.example`
- Test: `tests/unit/test_config_nlu.py`

- [ ] **Step 1: Failing tests**

```python
def test_intent_model_path_default_empty():
    from config import get_intent_model_path
    assert get_intent_model_path() == ""

def test_nlu_device_defaults_cpu():
    from config import get_nlu_device
    assert get_nlu_device() in ("cpu", "cuda")
```

Implement getters:

- `NLU_INTENT_MODEL_DIR` or `INTENT_MODEL_PATH` — **directory** with HF `config.json` + weights
- `NLU_NER_MODEL_DIR` / `NER_MODEL_PATH`
- `NLU_INTENT_MAX_CHARS` — optional cap on **intent model input** string after context shaping (for extremely long inputs; entities still use full raw message)
- `NLU_DEVICE` — `cuda` if available else `cpu` (or explicit `cpu` to force)
- `NLU_AUTOCAST` — `true`/`false` (FP16/BF16 on CUDA per design)

Wire names to existing getters where already present: **`get_intent_budget_ms()`** / **`get_entity_budget_ms()`** (from `INTENT_INFERENCE_BUDGET_MS` / `ENTITY_INFERENCE_BUDGET_MS` or design aliases `NLU_INTENT_BUDGET_MS` — align env names in `env.example` with `config.py` in this task).

- [ ] **Step 2: Run pytest** `tests/unit/test_config_nlu.py` — PASS

- [ ] **Step 3: Commit** — `feat(config): add NLU model paths and device`

---

### Task 4: `TransformerIntentEngine` (PhoBERT) with budget + load failure fallback

**Files:**

- Modify: `nlu/engines/utils.py`
- Modify: `nlu/engines/intent_transformer.py`
- Modify: `nlu/pipeline.py`

- [ ] **Step 1: Test with mocks** — patch `AutoModelForSequenceClassification.from_pretrained` to return an object whose `__call__` returns tensor logits; avoid network.

```python
def test_transformer_intent_uses_decide_intent(monkeypatch):
    # construct engine with fake model returning fixed logits
    ...
```

- [ ] **Step 2: Implement `run_with_budget_ms` in `utils.py`**

Complete the **Inference budget contract** section above; add `tests/unit/test_engine_budget.py` for timeout vs success paths **before** wiring transformers.

- [ ] **Step 3: Implement `TransformerIntentEngine`**

- Load tokenizer + model from `get_intent_model_path()`; if path missing or empty → pipeline must not crash: either skip constructing transformer and fall back to legacy, or engine returns `("fallback", 0.0)` on every call (choose one pattern and test it).
- Map `id2label` from `model.config` to label strings aligned with `data/intent.csv`.
- Wrap **entire** tokenize + forward + softmax in `run_with_budget_ms(..., budget_ms=get_intent_budget_ms(), on_timeout=("fallback", 0.0))` per contract.
- Apply optional **intent-input truncation** only to the string passed to the tokenizer (config key e.g. `NLU_INTENT_MAX_CHARS`), **after** `ContextProcessor` output — deterministic entities still use full raw `text` in `pipeline`.
- Use `get_nlu_device()` and optional autocast for CUDA.

- [ ] **Step 4: Wire `NLU_INTENT_ENGINE=transformer`** in `NLPPipeline` when path valid.

- [ ] **Step 5: Run** `pytest tests/unit/test_intent_transformer_engine.py tests/unit/test_pipeline_engines.py tests/unit/test_engine_budget.py -v`

- [ ] **Step 6: Commit** — `feat(nlu): PhoBERT transformer intent engine`

---

### Task 5: `TransformerNerEngine` — PhoBERT-family BIO + offsets on raw message

**Design trace:** `2026-04-03-intent-ner-models-v2-design.md` table row **Entities (neural)** + §**NER** + §**Offset contract**.

**Files:**

- Modify (verify): `nlu/engines/ner_transformer.py`
- Test: `tests/unit/test_ner_transformer_engine.py`
- Modify (verify): `nlu/pipeline.py`, `env.example`

**Model artifact expectation:** A directory compatible with:

```python
AutoTokenizer.from_pretrained(path, use_fast=True)
AutoModelForTokenClassification.from_pretrained(path)
```

where **`config.json`** defines `id2label` mapping to **BIO** tag strings. **Preferred backbone:** checkpoints trained from **`vinai/phobert-base`** (or **large-v2**) per design §Intent classifier / open questions.

- [ ] **Step 1: Pure function tests** for `offsets_to_spans` — BIO merge; Unicode code points (e.g. `"café"` spans).

- [ ] **Step 2: Artifact smoke test (local dir, no CI network)** — Add a test that builds a **tiny random** `PreTrainedModel` + tokenizer saved to `tmp_path`, loads `TransformerNerEngine(str(tmp_path))`, and asserts `extract("Hi")` returns a list (may be empty) **without** Hub download. Skip if engine already covered.

- [ ] **Step 3: Verify** `TransformerNerEngine.extract` implementation matches design:

  - **API:** `run_with_budget_ms(..., budget_ms=get_entity_budget_ms(), on_timeout=[])` around forward path.
  - **Tokenizer:** `return_offsets_mapping=True`, `truncation=True`, `max_length` capped (see `_max_token_length_for_model`).
  - **Raw message only:** no `ContextProcessor` / intent prefix on NER input.
  - **Truncation safety:** drop or clip spans so **`end` ≤ processed character prefix** (existing `processed_len` filter in engine).
  - **`NLU_NER_MAX_CHARS`:** if set, skip neural path for long `text` (`[]`).

- [ ] **Step 4: Verify merge** in `NLPPipeline.extract_entities`: **deterministic** always runs when `mode != off`; **transformer** adds non-overlapping `(label,start,end)` neural spans via `_merge_entity_lists`; neural entities use **`source="model"`**.

- [ ] **Step 5: Operator playbook (docs in `env.example` comment block)** — Document **Path A** (snapshot community PhoBERT NER to `NLU_NER_MODEL_DIR`) and **Path B** (fine-tune; see Task 11). Example env:

```env
NLU_ENTITY_ENGINE=transformer
NLU_NER_MODEL_DIR=models/ner/my-phobert-ner-export
```

- [ ] **Step 6: Commit** (if changes) — `docs(nlu): align NER task with PhoBERT artifact paths`

---

### Task 6: Long-input behavior (transformer + legacy)

**Files:**

- Modify: `nlu/pipeline.py`
- Create: `tests/unit/test_pipeline_long_input.py` (or extend `test_pipeline_edge_cases.py`)

**Spec:** Design doc §Edge cases + contract (truncate intent path, full-text deterministic entities, `score=top1` when inference succeeds).

- [ ] **Step 1: Failing tests** with `monkeypatch` / fake engines (same pipeline code path for **legacy and transformer** intent selection):

  - Build a message longer than intent truncation limit; assert `EntityExtractor` / deterministic path receives **full** `text` (patch `extract_entities` or spy).
  - Assert intent engine receives **truncated** `intent_input` only (when transformer + max length configured).
  - When fake intent returns `("fallback", 0.41)` with successful inference, assert `analysis["score"] == 0.41` and `analysis["intent"] == "fallback"`.

- [ ] **Step 2: Implement** truncation + wiring in `pipeline` if tests fail.

- [ ] **Step 3: Run** `pytest tests/unit/test_pipeline_long_input.py -v` (or equivalent path).

- [ ] **Step 4: Commit** — `test(nlu): long-input semantics for transformer path`

---

### Task 7: Observability (no raw text)

**Files:**

- Modify: `nlu/pipeline.py`, `nlu/engines/*.py`, optionally `main.py`

- [ ] **Step 1: Log** `engine`, `component`, `elapsed_ms`, `fallback_reason` — message length only, not content.

- [ ] **Step 2: Ensure** request id: if FastAPI middleware sets `request.state.request_id`, pass into `NLPService` / pipeline optional param — **or** document TODO for follow-up.

- [ ] **Step 3: Commit** — `feat(nlu): structured NLU logs without raw user text`

---

### Task 8: Integration test — schema + toggles

**Files:**

- Modify: `tests/integration/test_api.py` or new `tests/integration/test_nlu_engines.py`

- [ ] **Step 1:** `POST /chat/advanced` returns same JSON keys when env switches (use `monkeypatch` or test client with env).

- [ ] **Step 2: Commit** — `test(api): NLU schema stable under engine env`

---

### Task 9 (optional): Tier 1 `sticky_entities` in `ContextStore`

**Files:**

- Modify: `services/nlp_service.py`, `main.py`

Per design doc §Two-tier context: add `sticky_entities` dict to session context; update from last resolved entities. **Do not** block PhoBERT engines on this.

- [ ] **Step 1:** Unit test append/update behavior.

- [ ] **Step 2:** Commit — `feat(context): add sticky_entities map`

---

### Task 10a (optional): Training script — intent PhoBERT fine-tune

**Files:**

- Verify: `scripts/train_intent_phobert.py` (may already exist)

- [ ] **Step 1:** CLI: `--data data/intent.csv`, `--out models/intent/run1`, `--base vinai/phobert-base`.

- [ ] **Step 2:** Document in `env.example`: point **`NLU_INTENT_MODEL_DIR`** at exported dir.

- [ ] **Step 3:** Commit — `chore(scripts): document intent training artifact path` (only if docs drift)

---

### Task 11 (optional): Training script — PhoBERT **NER** fine-tune (Path B)

**Goal (design):** Produce a **`save_pretrained`** directory usable as **`NLU_NER_MODEL_DIR`**—same contract as **§PhoBERT-based NER artifacts** (BIO + `id2label` + fast tokenizer).

**Files:**

- Create: `scripts/train_ner_phobert.py`
- Create: `data/ner/README.md` (or comment in `env.example`) describing the **input format**
- Modify: `env.example` — NER training vars / example paths

- [ ] **Step 1: Choose on-disk training format (YAGNI)** — Use **JSON Lines**: one object per line, word-segmented:

```json
{"tokens": ["Đại", "học", "Xây", "dựng"], "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG"]}
```

Require: `len(tokens) == len(ner_tags)`; tags are **BIO**; include **`O`**.

- [ ] **Step 2: Failing smoke (optional)** — `tests/unit/test_train_ner_data_loader.py`: load two lines from a temp file; assert label set builds `label2id` / `id2label` with **`O` first** (typical HF convention):

```python
def test_ner_label_maps_include_o_first(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text(
        '{"tokens":["a","b"],"ner_tags":["B-X","O"]}\n'
        '{"tokens":["c"],"ner_tags":["O"]}\n',
        encoding="utf-8",
    )
    # import helper from scripts.train_ner_phobert or nlu.train.ner_data
    label2id, id2label = build_label_maps(p)
    assert id2label[0] == "O"
```

Implement **`build_label_maps`** in the script module (or `nlu/datasets/ner_jsonl.py` if you want importability).

- [ ] **Step 3: Implement `train_ner_phobert.py`** (mirror patterns from `scripts/train_intent_phobert.py`: `load_dotenv`, `TRAIN_*`-style env or CLI flags, GPU-friendly `DataLoader`, optional AMP).

  - Args: `--data` (jsonl path), `--out`, `--base` (default `vinai/phobert-base`), `--epochs`, `--batch-size`, `--lr`, `--max-len`.
  - Use **`AutoTokenizer.from_pretrained(base, use_fast=True)`** and **`AutoModelForTokenClassification.from_pretrained(base, num_labels=len(labels), id2label=..., label2id=...)`**.
  - **Subword alignment:** tokenize with `is_split_into_words=True` (word-level labels → subword label IDs; use `-100` for special/pad subwords in `labels` tensor per HF token classification docs).
  - Save: `model.save_pretrained(out)` + `tokenizer.save_pretrained(out)`.

- [ ] **Step 4: Manual run (developer machine, GPU)** —

```powershell
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; uv run python scripts/train_ner_phobert.py --data data/ner/train.jsonl --out models/ner/run1"
```

Expect: directory `models/ner/run1` loads in `TransformerNerEngine`.

- [ ] **Step 5: Commit** — `feat(scripts): add PhoBERT NER fine-tune entrypoint`

---

## Dependency / lockfile hygiene

- [ ] Search repo for `underthesea` imports; should only be comments. If `uv.lock` still lists `underthesea` without `pyproject` dependency, run `uv lock` refresh **or** document why lock includes extras.

## Verification (before claiming done)

```powershell
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; pytest -v"
pwsh -NoProfile -Command "cd 'c:\Users\ngoqh\Projects\nlp-chatbot'; ruff check ."
```

## Execution handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-04-03-intent-ner-models-v2.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — dispatch a fresh subagent per task; review between tasks (@superpowers:subagent-driven-development).

2. **Inline execution** — run tasks in this session with checkpoints (@superpowers:executing-plans).

**Which approach?**

**Note:** `docs/superpowers` may be gitignored; if `git add` fails, use `git add -f docs/superpowers/plans/2026-04-03-intent-ner-models-v2.md`.
