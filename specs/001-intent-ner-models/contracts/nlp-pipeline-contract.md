# Contract: NLP Pipeline (Intent + Entities)

This contract defines the stable interface that the rest of the application expects from the NLP layer.

## Public contract (pipeline output)

The pipeline MUST expose a method equivalent to:

```text
NLPPipeline.analyze(text: str) -> {"intent": str, "score": number, "entities": list}
```

Where:
- `intent`:
  - one of the canonical intent labels, OR `"fallback"`
- `score`:
  - numeric confidence in `[0.0, 1.0]`
  - represents the probability of the top predicted intent (`top1`)
  - on fallback due to missing/corrupted intent model artifacts: `score = 0.0`
- `entities`:
  - list of entities
  - each entity is `{label, text, start, end, source}`

## Fallback semantics (required)

Intent fallback triggers when:
- `top1 < T`, OR
- `(top1 - top2) < M`

When intent fallback triggers due to timeout/inference failure:
- return `intent="fallback"`
- `score=0.0` (and still return deterministic entities)

## Component-wise degradation semantics

- Intent failures MUST not crash entity extraction.
- Entity failures MUST not crash intent classification.
- Each component degrades independently to its specified fallback behavior.

## Logging requirements (operational contract)

- Logs MUST include request id (from middleware) and stage/component identifiers.
- Logs MUST avoid raw user text in production logs.

