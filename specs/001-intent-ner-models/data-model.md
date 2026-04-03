# Phase 1 Data Model: Model-based Intent + NER

## Public NLP Output (required to stay stable)

The pipeline MUST return:

```json
{"intent": "<string>", "score": <number>, "entities": [<Entity>]}
```

Where:
- `intent`: canonical intent label, or `"fallback"`
- `score`: confidence in `[0.0, 1.0]` representing the top intent probability `top1`
- `entities`: list of entity span objects

## Entity Span Object

Each entity item MUST be:

```json
{"label": "<string>", "text": "<string>", "start": <int>, "end": <int>, "source": "<string>"}
```

- `label`: canonical entity label
- `text`: extracted surface form (as seen in the original input)
- `start`: Unicode code point index (inclusive)
- `end`: Unicode code point index (exclusive)
- `source`: `"model"`, `"pattern"`, or `"dictionary"`

## Intent Label Set (from `data/intent.csv`)

Model-based intent classification MUST use the existing intent labels:
- `chao_hoi`
- `hoi_chi_tieu`
- `hoi_diem_chuan`
- `hoi_dieu_kien`
- `hoi_hoc_bong`
- `hoi_hoc_phi`
- `hoi_kenh_nop_ho_so`
- `hoi_nganh_hoc`
- `hoi_phuong_thuc`
- `hoi_thoi_gian_dk`
- `hoi_to_hop_mon`
- `reset_context`
- `tam_biet`
- `tro_giup`

## Entity Label Set (from `data/entity.json`)

NER tag set MUST map to these existing canonical entity labels:
- `CHI_TIEU`
- `CHUNG_CHI`
- `CHUYEN_NGANH`
- `DIEM_CHUAN`
- `DIEU_KIEN_TUYEN_SINH`
- `HOC_BONG`
- `HOC_PHI`
- `KY_THI`
- `MA_NGANH`
- `MON_HOC`
- `NAM_TUYEN_SINH`
- `PHUONG_THUC_TUYEN_SINH`
- `TEN_NGANH`
- `THOI_GIAN`
- `TO_HOP_MON`

## Inference Config (operator-controlled)

### Engine selection
- `INTENT_ENGINE`: `"model"` or `"legacy"`
- `ENTITY_ENGINE`: `"model"` or `"deterministic"`

### Fallback thresholds (operator-configurable)
- `T` (confidence threshold): default set via offline calibration (initial default in code may be derived from the existing `INTENT_THRESHOLD`)
- `M` (ambiguity margin): determines the `(top1 - top2)` margin fallback rule

### Time budgets
- `INTENT_INFERENCE_BUDGET_MS`: per-request/model timeout for intent inference
- `ENTITY_INFERENCE_BUDGET_MS`: per-request/model timeout for entity inference

## Model Artifacts (local filesystem)

The model inference components must load from local artifacts, for example:
- `model_artifacts/intent/` (sequence classification)
- `model_artifacts/ner/` (token classification)

At runtime:
- Missing/corrupted intent artifacts => `intent="fallback"`, `score=0.0`
- Missing/corrupted entity artifacts => deterministic patterns/dictionaries extraction, with `source` set to `"pattern"`/`"dictionary"`

## Interfaces (contract for implementation)

Implementations SHOULD follow these narrow contracts to keep model code mockable in tests:

### Intent model backend
- Input: `(text: str, synonym_map: dict[str, str] | None) -> probabilities`
- Output: ranked intent probabilities:
  - `top1_intent`, `top1_prob`
  - `top2_intent`, `top2_prob`

### NER model backend
- Input: `text: str`
- Output: list of predicted spans:
  - `label`, `text`, `start`, `end`

## Source Mapping for deterministic extraction

Deterministic extraction MUST preserve `source`:
- Pattern matches => `source="pattern"`
- Dictionary/phrase matches => `source="dictionary"`

If deterministic extraction also retains any additional rule-based component, the plan MUST still ensure `source` is set and spans include `start/end` in code points.

