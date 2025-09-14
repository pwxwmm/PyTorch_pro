# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Log Anomaly Demo (Training)

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from datasets import load_dataset

from model import create_model


def collate_fn(tokenizer):
    def _fn(batch):
        texts = [ex["text"] for ex in batch]
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.path.join("data", "train.txt"))
    parser.add_argument("--val", type=str, default=os.path.join("data", "val.txt"))
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", type=str, default=os.path.join("saved_model"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.train), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = load_dataset(
        "csv",
        data_files={"train": args.train, "validation": args.val},
        delimiter="\t",
        column_names=["label", "text"],
    )

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, num_labels=2).to(device)
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

        # simple validation loss
        model.eval()
        total = 0
        running = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                running += float(loss.item()) * batch["labels"].size(0)
                total += int(batch["labels"].size(0))
                pred = out.logits.argmax(dim=-1)
                correct += int((pred == batch["labels"]).sum().item())
        val_loss = running / max(1, total)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved model to {args.save_dir}")


if __name__ == "__main__":
    main()
