# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Log Anomaly Demo (Inference)

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_saved(save_dir: str = "saved_model"):
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model.eval()
    return model, tokenizer


def predict_log(text: str, save_dir: str = "saved_model") -> str:
    model, tokenizer = load_saved(save_dir)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return "异常" if probs[0][1] > 0.5 else "正常"


if __name__ == "__main__":
    print(predict_log("ERROR Failed to allocate GPU memory"))
