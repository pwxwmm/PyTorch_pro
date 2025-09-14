# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Training Failure RCA (Inference)

import json
import os
from typing import Dict, List

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_saved(
    save_dir: str = "saved_model",
) -> tuple[BertForSequenceClassification, BertTokenizer, List[str], Dict[str, str]]:
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_space.json"), "r", encoding="utf-8") as f:
        label_meta = json.load(f)
    labels: List[str] = label_meta["labels"]

    # Optional actions mapping from repo root labels.json if present in save_dir's parent
    actions: Dict[str, str] = {}
    repo_labels = os.path.join(os.path.dirname(save_dir), "labels.json")
    if os.path.exists(repo_labels):
        with open(repo_labels, "r", encoding="utf-8") as f:
            raw = json.load(f)
            actions = raw.get("actions", {})

    model.eval()
    return model, tokenizer, labels, actions


def predict_rca(
    text: str, save_dir: str = "saved_model", threshold: float = 0.5
) -> Dict[str, float]:
    model, tokenizer, labels, _ = load_saved(save_dir)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=160
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze(0)
    result = {
        labels[i]: float(probs[i])
        for i in range(len(labels))
        if float(probs[i]) >= threshold
    }
    return result


def explain_actions(
    predicted: Dict[str, float], save_dir: str = "saved_model"
) -> Dict[str, str]:
    _, _, _, actions = load_saved(save_dir)
    return {k: actions.get(k, "") for k in predicted.keys()}


if __name__ == "__main__":
    text = "2025-09-10 12:30:10 ERROR RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    pred = predict_rca(text)
    print("Predicted root causes:", pred)
    print("Suggested actions:", explain_actions(pred))
