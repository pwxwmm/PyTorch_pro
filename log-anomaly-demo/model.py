# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Log Anomaly Demo (Model)

from transformers import BertForSequenceClassification


def create_model(
    model_name: str = "bert-base-chinese", num_labels: int = 2
) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
