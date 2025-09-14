# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Training Failure Root-Cause Analysis (Multi-label)

from transformers import BertForSequenceClassification


def create_multilabel_model(
    model_name: str = "bert-base-chinese", num_labels: int = 8
) -> BertForSequenceClassification:
    # problem_type set to multi_label_classification makes HF compute BCEWithLogitsLoss
    return BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
