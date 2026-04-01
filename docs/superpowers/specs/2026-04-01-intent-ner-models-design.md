# Intent + NER model migration design (GPU local)

Date: 2026-04-01  
Repo: `nlp-chatbot`  
Scope: Replace TF‑IDF + cosine intent matching with a fine-tuned model, and replace `underthesea` tokenization + NER with more efficient model-based components (GPU local).

## Goals

- Improve intent detection quality vs TF‑IDF centroid cosine.
- Replace `underthesea` usage (tokenization + NER) with model-based equivalents suitable for **local GPU** runtime.
- Preserve the current API shape and system behavior:
  - `NLPPipeline.analyze(text) -> {"intent", "score", "entities"}`
  - intent confidence threshold with `"fallback"` intent on low confidence
  - keep existing dictionary + pattern entity extraction (domain-specific and fast)

## Non-goals

- No new UI work.
- No migration of CSV data into a database.
- No multi-language product scope expansion (models may be multilingual, but UX/spec remains Vietnamese-first).
- No custom NER training in this iteration (use an existing pretrained NER model).

## Current baseline (for reference)

- Intent detection: `nlu/intent.py` implements TF‑IDF vectors + centroid per intent, cosine similarity at runtime.
- Vietnamese preprocessing: `nlu/preprocess.py` uses `underthesea.word_tokenize` when available.
- NER: `nlu/entities.py` optionally uses `underthesea.ner`; entities also extracted via patterns and dictionary phrase matching.

## Proposed architecture

### Overview

1) **Intent classifier** (fine-tuned Transformer encoder + classification head) replaces TF‑IDF/cosine.  
2) **Transformer NER** replaces `underthesea.ner`.  
3) **Lightweight normalization remains** for dictionary/pattern entities and synonyms; intent no longer depends on Vietnamese word tokenization.

```
User text
  ├─ normalize_text (keep; NFC + cleanup)
  ├─ IntentClassifier.detect(text)  -> (intent, confidence)
  └─ EntityExtractor.extract(text)  -> patterns + dictionary + transformer NER
```

### Module boundaries

- `nlu/intent_model/` (new)
  - Training script(s) + model loading utilities
  - Runtime `IntentClassifier` (drop-in replacement for `IntentDetector` usage in `NLPPipeline`)
- `nlu/entities.py` (update)
  - Keep pattern + dictionary logic intact
  - Replace `_extract_by_ner()` implementation with Transformer NER pipeline
- `nlu/preprocess.py` (update)
  - Remove reliance on `underthesea` for intent flow
  - Keep `normalize_text()` and synonym mapping helper(s)
- `nlu/pipeline.py` (update)
  - Wire new intent classifier; keep output contract

## Intent detection design (fine-tuned classifier)

### Data

Source: `data/intent.csv` with at least:

- `utterance`: user-style text
- `intent`: label

We will reuse the existing examples; no manual labeling changes are required for the first iteration.

### Model choice

Primary recommendation (Vietnamese-first):

- Encoder: PhoBERT-family (or equivalent Vietnamese encoder available via `transformers`)
- Classification head: linear layer over pooled representation

Rationale:

- Subword tokenization and contextual embeddings remove the need for explicit Vietnamese word segmentation.
- Fine-tuned classifiers typically outperform nearest-neighbor TF‑IDF and embedding retrieval on close/overlapping intents.

### Runtime API

Expose a small interface:

- `detect(text: str) -> tuple[str, float]`
  - returns `(intent, confidence)`
  - if `confidence < intent_threshold`: return `("fallback", confidence)`

Confidence definition:

- Softmax probability of the predicted class.

### Preprocessing for intent

- Keep `normalize_text()` (unicode NFC + punctuation cleanup).
- Do **not** apply `underthesea.word_tokenize`.
- Optional: apply synonym normalization as a text-level substitution only if it demonstrably helps intent accuracy; otherwise keep synonym mapping confined to entity extraction.

### Training and artifacts

Add a training entrypoint that:

- Reads `data/intent.csv`
- Splits train/val (stratified by label if feasible)
- Fine-tunes the classifier on GPU if available
- Exports an artifact directory with:
  - model weights
  - tokenizer files
  - label mapping (intent ↔ id)
  - training metadata (dataset hash, commit hash, date)

Artifact location (proposal):

- `models/intent/<yyyy-mm-dd>_<model-name>/...`

Loading behavior:

- Backend loads model on startup; warmup optional.
- Device selection: GPU when present; otherwise CPU (still functional, slower).

### Performance targets (informal)

- Inference: < ~50ms per request on a modest GPU for single-sentence classification (excluding NER).
- Startup: acceptable to take seconds due to model load.

## Entity extraction design (patterns + dictionary + Transformer NER)

### Keep existing deterministic extraction

Keep:

- `_extract_by_patterns()`
- `_extract_by_dictionaries()`
- label aliasing / dedup logic

Reasoning:

- These rules encode domain knowledge (HUCE admissions) and provide predictable coverage.

### Replace `underthesea.ner` with Transformer NER

Implement `_extract_by_ner(text)` using a pretrained NER model:

- Use `transformers` token classification pipeline (or an equivalent efficient runtime).
- Convert model outputs into the current internal format:
  - `{"label": <tag>, "text": <span>, "source": "ner"}`

Label policy:

- Preserve current label alias mapping behavior in `EntityExtractor` (e.g., `NAM` → `NAM_HOC`).
- If the chosen NER model uses generic labels (PER/LOC/ORG), either:
  - keep them as-is (low coupling), or
  - map them into a small internal label set **only if** it improves downstream behaviors.

Span policy:

- Merge subword tokens into a clean surface span.
- Dedup with existing pattern/dictionary spans as currently done.

### Tokenization strategy

- Dictionary/pattern matching continues to use `normalize_text()` and substring checks; no word segmentation required.
- If phrase boundary issues appear, we can optionally add a lightweight Vietnamese segmenter later, but it is out of scope for this first migration.

## Configuration

Add configuration surface (via `config.py` / env vars) for:

- Intent model path or model name
- NER model name
- Device selection (`auto|cpu|cuda`)
- Intent threshold (already present)
- Batch size / max length (optional)

## Error handling & fallbacks

- If model load fails:
  - Intent: fail closed to `"fallback"` intent with score 0.0 (and log clearly).
  - NER: skip transformer NER but keep pattern/dictionary extraction.
- If GPU unavailable:
  - run on CPU; keep behavior correct, slower.

## Testing strategy

- Unit tests:
  - Intent detector returns stable types and respects threshold/fallback.
  - Entity extractor still returns deterministic pattern/dictionary entities.
  - NER adapter merges tokens into spans and produces expected schema.
- Integration tests:
  - `POST /chat/advanced` still responds with intent + entities fields.
- Non-determinism guard:
  - For unit tests, mock model inference outputs rather than depending on downloaded weights.

## Rollout / migration plan (high level)

- Phase 1: Introduce new modules + feature-flag the classifier/NER (default on in dev).
- Phase 2: Remove `underthesea` dependency once transformer alternatives are stable.
- Phase 3 (optional): Add hybrid fallback (embedding retrieval) for low-confidence intents.

## Open questions (tracked for implementation plan)

- Exact model selection:
  - Which PhoBERT checkpoint (or alternative) for intent?
  - Which Vietnamese NER checkpoint for NER?
- Storage:
  - Do we commit model artifacts to git, or download on first run / cache locally?
- Deployment:
  - Target OS/driver constraints for GPU runtime in production.

