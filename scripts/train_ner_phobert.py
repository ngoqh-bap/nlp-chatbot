#!/usr/bin/env python3
"""Fine-tune PhoBERT for token classification (BIO NER) from JSON Lines data.

Each input line is a JSON object with word-level ``tokens`` and ``ner_tags`` (see ``data/ner/README.md``).

Example:
  uv run python scripts/train_ner_phobert.py --data data/ner/train.jsonl --out models/ner/run1

Reuses the same ``TRAIN_*`` env defaults as ``train_intent_phobert.py`` where applicable.
Point ``NLU_NER_MODEL_DIR`` at ``--out`` when using ``NLU_ENTITY_ENGINE=transformer``.
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from functools import partial
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from nlu.datasets.ner_jsonl import build_label_maps_from_rows, load_ner_jsonl


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v else default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(explicit: str) -> torch.device:
    explicit = explicit.lower()
    if explicit == "cpu":
        return torch.device("cpu")
    if explicit == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("TRAIN_DEVICE=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_amp(
    device: torch.device,
    amp_mode: str,
) -> tuple[bool, torch.dtype | None, bool]:
    if device.type != "cuda":
        return False, None, False
    m = amp_mode.lower().strip()
    if m in ("none", "off", "fp32", "no"):
        return False, None, False
    if m == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        print("Warning: TRAIN_AMP=bf16 but BF16 not supported; falling back to fp16 + GradScaler.")
        return True, torch.float16, True
    return True, torch.float16, True


class _NerWordDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[list[str], list[str]]],
        tokenizer,
        label2id: dict[str, int],
        max_len: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        tokens, tags = self.rows[i]
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        word_ids = enc.word_ids()
        label_ids: list[int] = []
        previous_word_idx: int | None = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label_ids,
        }


def _collate_ner_batch(batch: list[dict], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for x in batch:
        pad_n = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_n)
        attention_mask.append(x["attention_mask"] + [0] * pad_n)
        labels.append(x["labels"] + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    default_batch = _env_int("TRAIN_BATCH_SIZE", 16 if torch.cuda.is_available() else 8)
    ap = argparse.ArgumentParser(description="Fine-tune PhoBERT for BIO NER (token classification)")
    ap.add_argument("--data", type=Path, required=True, help="JSON Lines: tokens + ner_tags per line")
    ap.add_argument("--out", type=Path, required=True, help="Directory to save tokenizer + model")
    ap.add_argument("--base", default=_env_str("TRAIN_BASE_MODEL", "vinai/phobert-base"))
    ap.add_argument("--epochs", type=int, default=_env_int("TRAIN_EPOCHS", 5))
    ap.add_argument("--batch-size", type=int, default=default_batch)
    ap.add_argument("--grad-accum", type=int, default=_env_int("TRAIN_GRAD_ACCUM", 1))
    ap.add_argument("--lr", type=float, default=_env_float("TRAIN_LR", 2e-5))
    ap.add_argument("--max-len", type=int, default=_env_int("TRAIN_MAX_LEN", 128))
    ap.add_argument("--warmup-ratio", type=float, default=_env_float("TRAIN_WARMUP_RATIO", 0.05))
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=_env_str("TRAIN_DEVICE", "auto"),
    )
    ap.add_argument(
        "--amp",
        choices=("bf16", "fp16", "none", "auto"),
        default=_env_str("TRAIN_AMP", "auto"),
    )
    ap.add_argument("--num-workers", type=int, default=_env_int("TRAIN_NUM_WORKERS", 0))
    ap.add_argument("--seed", type=int, default=_env_int("TRAIN_SEED", 42))
    ap.add_argument("--max-grad-norm", type=float, default=_env_float("TRAIN_MAX_GRAD_NORM", 1.0))
    args = ap.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r".*[Uu]nauthenticated requests to the HF Hub.*",
    )

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_mode = args.amp
    if amp_mode == "auto" and device.type == "cuda":
        amp_mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    use_autocast, amp_dtype, use_scaler = _resolve_amp(device, amp_mode if device.type == "cuda" else "none")
    scaler: torch.amp.GradScaler | None = None
    if device.type == "cuda" and use_scaler:
        scaler = torch.amp.GradScaler("cuda")

    rows = load_ner_jsonl(args.data)
    label2id, id2label = build_label_maps_from_rows(rows)
    n_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForTokenClassification.from_pretrained(
        args.base,
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    ds = _NerWordDataset(rows, tokenizer, label2id, args.max_len)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise SystemExit("Tokenizer has no pad_token_id; set tokenizer.pad_token.")
    collate_fn = partial(_collate_ner_batch, pad_token_id=int(pad_id))

    pin = device.type == "cuda"
    nw = max(0, args.num_workers)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        collate_fn=collate_fn,
    )

    steps = len(loader) * args.epochs
    warmup = int(steps * args.warmup_ratio)
    optim = AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, warmup, steps)

    model.train()
    optim.zero_grad(set_to_none=True)
    accum = max(1, args.grad_accum)

    def _optimizer_step(*, partial_accum: int | None = None) -> None:
        if scaler is not None:
            scaler.unscale_(optim)
        if partial_accum is not None and 0 < partial_accum < accum:
            scale = accum / partial_accum
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pending = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
            if use_autocast and amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    out = model(**batch)
                    raw = out.loss
            else:
                out = model(**batch)
                raw = out.loss

            loss = raw / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            pending += 1
            epoch_loss += float(raw.detach().item())
            n_batches += 1

            if pending >= accum:
                _optimizer_step()
                pending = 0

        if pending > 0:
            _optimizer_step(partial_accum=pending)

        print(f"epoch {epoch + 1}/{args.epochs} mean_loss={epoch_loss / max(1, n_batches):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to {args.out.resolve()} — set NLU_NER_MODEL_DIR to this path.")


if __name__ == "__main__":
    main()
