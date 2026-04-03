#!/usr/bin/env python3
"""Fine-tune a PhoBERT sequence classifier from data/intent.csv (utterance, intent columns).

Example:
  uv run python scripts/train_intent_phobert.py --data data/intent.csv --out models/intent/run1

Point NLU_INTENT_MODEL_DIR at ``--out`` after training when using NLU_INTENT_ENGINE=transformer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class _IntentRows(Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer, label2id: dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[self.labels[i]], dtype=torch.long)
        return item


def _load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("utterance") or "").strip()
            intent = (row.get("intent") or "").strip()
            if u and intent:
                texts.append(u)
                labels.append(intent)
    if not texts:
        raise SystemExit(f"No rows in {path}")
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune PhoBERT for intent classification")
    ap.add_argument("--data", type=Path, default=Path("data/intent.csv"))
    ap.add_argument("--out", type=Path, required=True, help="Directory to save tokenizer + model")
    ap.add_argument("--base", default="vinai/phobert-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    args = ap.parse_args()

    texts, labels = _load_csv(args.data)
    uniq = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(uniq),
        id2label=id2label,
        label2id=label2id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = _IntentRows(texts, labels, tokenizer, label2id, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    steps = len(loader) * args.epochs
    warmup = int(steps * args.warmup_ratio)
    optim = AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, warmup, steps)

    model.train()
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to {args.out.resolve()} — set NLU_INTENT_MODEL_DIR to this path.")


if __name__ == "__main__":
    main()
