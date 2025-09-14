# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Training Failure RCA (FastAPI)

from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_rca, explain_actions


class RCAInput(BaseModel):
    text: str
    threshold: float = 0.5
    save_dir: str = "saved_model"


app = FastAPI(title="Train Failure RCA API", version="0.1.0")


@app.post("/predict")
async def predict(inp: RCAInput) -> Dict[str, Dict[str, float]]:
    probs = predict_rca(inp.text, inp.save_dir, inp.threshold)
    actions = explain_actions(probs, inp.save_dir)
    return {"probabilities": probs, "actions": actions}
