# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Training Failure RCA (Multi-label BERT)

import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from datasets import load_dataset

from model import create_multilabel_model


def parse_labels(labels_str: str, all_labels: List[str]) -> torch.Tensor:
    # labels are comma-separated, e.g. "CUDA_OutOfMemory,OOM"
    vec = torch.zeros(len(all_labels), dtype=torch.float32)
    if isinstance(labels_str, str) and labels_str.strip():
        for name in [x.strip() for x in labels_str.split(",") if x.strip()]:
            if name in all_labels:
                vec[all_labels.index(name)] = 1.0
    return vec


def collate_fn(tokenizer, all_labels: List[str]):
    def _fn(batch):
        texts = [ex["text"] for ex in batch]
        labels = torch.stack([parse_labels(ex["labels"], all_labels) for ex in batch])
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.path.join("data", "train.tsv"))
    parser.add_argument("--val", type=str, default=os.path.join("data", "val.tsv"))
    parser.add_argument("--labels", type=str, default=os.path.join("labels.json"))
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default=os.path.join("saved_model"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.train), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.labels, "r", encoding="utf-8") as f:
        meta = json.load(f)
    all_labels: List[str] = meta["labels"]

    dataset = load_dataset(
        "csv",
        data_files={"train": args.train, "validation": args.val},
        delimiter="\t",
        column_names=["labels", "text"],
    )

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, all_labels),
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, all_labels),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_multilabel_model(args.model_name, num_labels=len(all_labels)).to(
        device
    )
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch["labels"].size(0)
            total += int(batch["labels"].size(0))
        print(f"Epoch {epoch} train_loss={running/max(1,total):.6f}")

        model.eval()
        total = 0
        running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                running += float(loss.item()) * batch["labels"].size(0)
                total += int(batch["labels"].size(0))
        val_loss = running / max(1, total)
        print(f"Epoch {epoch} val_loss={val_loss:.6f}")

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    # Save label mapping for inference
    with open(
        os.path.join(args.save_dir, "label_space.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=False, indent=2)
    print(f"Saved model to {args.save_dir}")


if __name__ == "__main__":
    main()
