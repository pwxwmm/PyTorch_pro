# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Metric Forecast Demo (Inference)

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from model import LSTMForecast


def load_model(save_dir: str) -> Tuple[LSTMForecast, dict]:
    meta_path = os.path.join(save_dir, "metadata.json")
    ckpt_path = os.path.join(save_dir, "forecast.pt")
    if not (os.path.exists(meta_path) and os.path.exists(ckpt_path)):
        raise FileNotFoundError("Model files not found. Train first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = LSTMForecast(
        input_size=1,
        hidden_size=meta["model"]["hidden_size"],
        num_layers=meta["model"]["num_layers"],
        dropout=meta["model"]["dropout"],
        output_size=1,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])  # type: ignore[index]
    model.eval()
    return model, meta


def zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def inverse_zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def forecast_next_from_series(series: List[float], save_dir: str) -> float:
    model, meta = load_model(save_dir)
    seq_len = int(meta["seq_len"])
    if len(series) < seq_len:
        raise ValueError(f"Need at least {seq_len} points, got {len(series)}")

    window = np.asarray(series[-seq_len:], dtype=np.float32)
    window_norm = zscore(window, float(meta["mean"]), float(meta["std"]))
    x = torch.from_numpy(window_norm).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    with torch.no_grad():
        y_hat_norm = model(x).squeeze(0).numpy()  # (1,) -> (1,)
    y_hat = inverse_zscore(y_hat_norm, float(meta["mean"]), float(meta["std"]))
    return float(y_hat.squeeze())


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv", type=str, default=os.path.join("data", "metrics.csv")
    )
    parser.add_argument("--save_dir", type=str, default=os.path.join("saved_model"))
    args = parser.parse_args()

    # Load last window from CSV
    df = pd.read_csv(args.data_csv)
    values = (
        df["value"].to_numpy(dtype=np.float32)
        if "value" in df
        else df.iloc[:, -1].to_numpy(dtype=np.float32)
    )
    model, meta = load_model(args.save_dir)
    seq_len = int(meta["seq_len"])
    pred = forecast_next_from_series(values[-seq_len:].tolist(), args.save_dir)
    print(f"Next value prediction: {pred:.6f}")


if __name__ == "__main__":
    cli_main()
