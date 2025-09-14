# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Log Anomaly Demo (FastAPI)

from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_log


class PredictInput(BaseModel):
    text: str
    save_dir: str = "saved_model"


app = FastAPI(title="Log Anomaly API", version="0.1.0")


@app.post("/predict")
async def predict(inp: PredictInput):
    label = predict_log(inp.text, inp.save_dir)
    return {"text": inp.text, "result": label}
