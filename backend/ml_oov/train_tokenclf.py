"""Train KoELECTRA token classification model (CPU-friendly defaults).

Example:
  python -m backend.ml_oov.train_tokenclf \
    --tokenizer monologg/koelectra-base-v3-discriminator \
    --train backend/data/oov_gold/train.jsonl \
    --dev backend/data/oov_gold/dev.jsonl \
    --out models/koelectra_oov \
    --max_length 128 --epochs 2 --batch 4 --cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .labels import LABELS, LABEL2ID, ID2LABEL
from .dataset_builder import build_features

def _load_features(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cache_dir = Path("backend/ml_oov/_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat = cache_dir / "train.features.jsonl"
    dev_feat = cache_dir / "dev.features.jsonl"

    build_features(args.train, train_feat, tokenizer_name_or_path=args.tokenizer, max_length=args.max_length)
    build_features(args.dev, dev_feat, tokenizer_name_or_path=args.tokenizer, max_length=args.max_length)

    train_rows = _load_features(train_feat)
    dev_rows = _load_features(dev_feat)

    from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, set_seed
    import torch

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.tokenizer,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class JsonlDataset(torch.utils.data.Dataset):
        def __init__(self, rows: List[dict]):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            return {
                "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(r["labels"], dtype=torch.long),
            }

    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=0,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=JsonlDataset(train_rows),
        eval_dataset=JsonlDataset(dev_rows),
        tokenizer=tok,
    )

    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))

    cfg = {
        "tokenizer_or_base_model": args.tokenizer,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "labels": LABELS,
    }
    (args.out / "training_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_tokenclf] saved checkpoint to {args.out}")

if __name__ == "__main__":
    main()
