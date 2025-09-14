# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Metric Forecast Demo (FastAPI)

from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import forecast_next_from_series


class PredictRequest(BaseModel):
    series: List[float]
    save_dir: Optional[str] = os.path.join("saved_model")


app = FastAPI(title="Metric Forecast API", version="0.1.0")


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        value = forecast_next_from_series(req.series, req.save_dir)
        return {"next_value": float(value)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
