# NER training data (JSON Lines)

Used by `scripts/train_ner_phobert.py`. One JSON object per line, **word-segmented**:

```json
{"tokens": ["Đại", "học", "Xây", "dựng"], "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG"]}
```

Rules:

- `len(tokens) == len(ner_tags)` on every line.
- Tags are **BIO** strings and every corpus must include at least one **`O`** tag.
- **Id 0** is always `O` in the saved model.

Train:

```bash
uv run python scripts/train_ner_phobert.py --data data/ner/train.jsonl --out models/ner/run1
```

Then set `NLU_NER_MODEL_DIR=models/ner/run1` and `NLU_ENTITY_ENGINE=transformer`.

See `env.example` for **Path A** (download a community PhoBERT NER checkpoint) vs **Path B** (this script).
